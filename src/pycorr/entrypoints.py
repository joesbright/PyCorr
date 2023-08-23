import os, glob, argparse, yaml, logging, time

import tomli as tomllib # `tomllib` as of Python 3.11 (PEP 680)
import h5py
import numpy
from guppi.guppi import Guppi
import pyproj

import pycorr

def _degrees_process(value):
    if isinstance(value, str):
        if value.count(':') == 2:
            value = value.split(':')
            value_f = float(value[0])
            if value_f < 0:
                value_f -= (float(value[1]) + float(value[2])/60)/60
            else:
                value_f += (float(value[1]) + float(value[2])/60)/60
            return value_f
        return float(value)
    return float(value)

def transform_antenna_positions_ecef_to_xyz(longitude_deg, latitude_deg, altitude, antenna_positions):
    transformer = pyproj.Proj.from_proj(
        pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84'),
        pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84'),
    )
    telescopeCenterXyz = transformer.transform(
        longitude_deg,
        latitude_deg,
        altitude,
    )
    for i in range(antenna_positions.shape[0]):
        antenna_positions[i, :] -= telescopeCenterXyz

def _get_telescope_metadata(telescope_info_filepath):
    """
    Returns a standardised formation of the TOML/YAML/BFR5 contents:
    {
        "telescope_name": str,
        "longitude": float, # radians
        "latitude": float, # radians
        "altitude": float,
        "antenna_position_frame": "xyz", # metres relative to lla
        "antennas": [
            {
                "name": str,
                "position": [X, Y, Z],
                "number": int,
                "diameter": float,
            }
        ]
    }
    """
    _, telinfo_ext = os.path.splitext(telescope_info_filepath)
    if telinfo_ext in [".toml"]:
        with open(telescope_info_filepath, mode="rb") as f:
            telescope_info = tomllib.load(f)
    elif telinfo_ext in [".yaml", ".yml"]:
        with open(telescope_info_filepath, mode="r") as f:
            telescope_info = yaml.load(f, Loader=yaml.CSafeLoader)
    elif telinfo_ext in [".bfr5"]:
        with h5py.File(bfr5_file, 'r') as f:
            telescope_info = {
                "telescope_name": f["telinfo"]["telescope_name"][()],
                "longitude": f["telinfo"]["longitude"][()],
                "latitude": f["telinfo"]["latitude"][()],
                "altitude": f["telinfo"]["altitude"][()],
                "antenna_position_frame": f["telinfo"]["antenna_position_frame"][()],
                "antennas": [
                    {
                        "number": antenna_number,
                        "name": f["telinfo"]["antenna_names"][i],
                        "position": f["telinfo"]["antenna_positions"][i],
                        "diameter": f["telinfo"]["antenna_diameters"][i],
                    }
                    for i, antenna_number in enumerate(f["telinfo"]["antenna_numbers"])
                ]
            }
    else:
        raise ValueError(f"Unknown file format '{telinfo_ext}' ({os.path.basename(telescope_info_filepath)}). Known formats are: yaml, toml.")

    longitude = _degrees_process(telescope_info["longitude"])
    latitude = _degrees_process(telescope_info["latitude"])
    altitude = telescope_info["altitude"]
    antenna_positions = numpy.array([antenna["position"] for antenna in telescope_info["antennas"]])

    # TODO handle enu
    telinfo_antposframe = telescope_info["antenna_position_frame"].lower()
    assert telinfo_antposframe in ["xyz", "ecef"]

    if telinfo_antposframe == "ecef":
        transform_antenna_positions_ecef_to_xyz(
            longitude,
            latitude,
            altitude,
            antenna_positions
        )

    return {
        "telescope_name": telescope_info["telescope_name"],
        "longitude": longitude*numpy.pi/180.0,
        "latitude": latitude*numpy.pi/180.0,
        "altitude": altitude,
        "antenna_position_frame": "xyz",
        "antennas": [
            {
                "name": ant_info["name"],
                "position": antenna_positions[ant_enum],
                "number": int(ant_info.get("number", ant_enum)),
                "diameter": ant_info.get("diameter", telescope_info["antenna_diameter"]),
            }
            for ant_enum, ant_info in enumerate(telescope_info["antennas"])
        ]
    }

def filter_telinfo(
    telinfo,
    guppi_header
):
    antenna_names = []
    guppi_is_ata = guppi_header.get("TELESCOP", "UNKNOWN") == "ATA"
    if guppi_is_ata:
        pycorr.logger.warning("GUPPI file appears to be from the ATA, will drop the last character from the listed antenna names.")
    for i in range(100):
        key = f"ANTNMS{i:02d}"
        if key not in guppi_header:
            break

        if guppi_is_ata:
            # ATA antenna names have LO suffixed
            antenna_names += map(lambda name: name[:-1], guppi_header[key].split(","))
        else:
            antenna_names += guppi_header[key].split(",")
    
    nants = guppi_header.get("NANTS", 1)

    if len(antenna_names) > 0:
        antenna_names = antenna_names[:nants]
        antenna_telinfo = {
            antenna["name"]: antenna
            for antenna in telinfo["antennas"]
            if antenna["name"] in antenna_names
        }
        assert len(antenna_telinfo) == len(antenna_names), f"Telescope information does not cover RAW listed antenna: {set(antenna_names).difference(set([ant['name'] for ant in telinfo['antennas']]))}"
    else:
        pycorr.logger.warning("No antenna names listed in the GUPPI header under 'ANTNMS%d{2}' entries. Using all provided antenna, sorted by number.")
        antenna_number_name_map = {
            antenna["number"]: antenna["name"]
            for antenna in telinfo["antennas"]
        }
        antenna_numbers = list(antenna_number_name_map.keys())
        antenna_numbers.sort()
        antenna_names = [
            antenna_number_name_map[antnum]
            for antnum in antenna_numbers
        ]

    
    return {
        "telescope_name": telinfo["telescope_name"],
        "longitude": telinfo["longitude"],
        "latitude": telinfo["latitude"],
        "altitude": telinfo["altitude"],
        "antenna_position_frame": telinfo["antenna_position_frame"],
        "antennas": [
            antenna_telinfo[antname]
            for antname in antenna_names
        ]
    }

def _get_jd(
    tbin,
    sampleperblk,
    piperblk,
    synctime,
    pktidx
):
    unix = synctime + (pktidx * (sampleperblk/piperblk)) * tbin
    return 2440587.5 + unix/86400

def main():
    
    parser = argparse.ArgumentParser(
        description="Correlate the data of a RAW file set, producing a UVH5 file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "raw_filepaths",
        type=str,
        nargs="+",
        help="The path to the GUPPI RAW file stem or of all files.",
    )
    parser.add_argument(
        "-t",
        "--telescope-info-filepath",
        type=str,
        required=True,
        help="The path to telescope information (YAML/TOML or BFR5['telinfo']).",
    )
    parser.add_argument(
        "-u",
        "--upchannelisation-rate",
        type=int,
        default=1,
        help="The upchannelisation rate.",
    )
    parser.add_argument(
        "-i",
        "--integration-rate",
        type=int,
        default=1,
        help="The integration rate.",
    )
    parser.add_argument(
        "-p",
        "--polarisations",
        type=str,
        default=None,
        help="The polarisation characters for each polarisation, as a string (e.g. 'xy').",
    )
    parser.add_argument(
        "-b", "--frequency-mhz-begin",
        type=float,
        default=None,
        help="The lowest frequency (MHz) of the fine-spectra to analyse (at least 1 channel will be processed).",
    )
    parser.add_argument(
        "-e", "--frequency-mhz-end",
        type=float,
        default=None,
        help="The highest frequency (MHz) of the fine-spectra to analyse (at least 1 channel will be processed).",
    )
    parser.add_argument(
        "--output-filepath",
        type=str,
        default=None,
        help="The path to which the output will be written (instead of alongside the raw_filepath).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase the verbosity of the generation (0=Error, 1=Warn, 2=Info (progress statements), 3=Debug)."
    )

    args = parser.parse_args()
    pycorr.logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )

    datablock_time_requirement = args.upchannelisation_rate

    telinfo = _get_telescope_metadata(args.telescope_info_filepath)
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        pycorr.logger.info(f"Given RAW filepath does not exist, assuming it is the stem.")
        args.raw_filepaths = glob.glob(f"{args.raw_filepaths[0]}*.raw")
        pycorr.logger.info(f"Found {args.raw_filepaths}.")
    
    input_dir, input_filename = os.path.split(args.raw_filepaths[0])
    if args.output_filepath is None:
        output_filepath = os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.uvh5")
    else:
        output_filepath = args.output_filepath

    pycorr.logger.debug(f"GUPPI RAW files: {args.raw_filepaths}")
    guppi_bytes_total = numpy.sum(list(map(os.path.getsize, args.raw_filepaths)))
    pycorr.logger.debug(f"Total GUPPI RAW bytes: {guppi_bytes_total/(10**6)} MB")

    guppi = Guppi(args.raw_filepaths.pop(0))
    guppi_header, guppi_data = guppi.read_next_block()
    telinfo = filter_telinfo(telinfo, guppi_header)

    ant_1_array, ant_2_array = pycorr.get_uvh5_ant_arrays(telinfo["antennas"])
    num_bls = len(ant_1_array)

    nants = guppi_header.get("NANTS", 1)
    npol = guppi_header["NPOL"]
    nchan = guppi_header["OBSNCHAN"] // nants
    ntimes = guppi_data.shape[2]
    schan = guppi_header.get("SCHAN", 0)
    coarse_chan_bw = guppi_header["CHAN_BW"]
    frequency_channel_0_mhz = guppi_header["OBSFREQ"] - (nchan/2 + 0.5)*coarse_chan_bw

    upchan_bw = coarse_chan_bw/args.upchannelisation_rate
    frequencies_mhz = frequency_channel_0_mhz + numpy.arange(nchan*args.upchannelisation_rate)*upchan_bw

    frequency_lowest_mhz = min(frequencies_mhz[0], frequencies_mhz[-1])
    frequency_highest_mhz = max(frequencies_mhz[0], frequencies_mhz[-1])
    if args.frequency_mhz_begin is None:
        args.frequency_mhz_begin = frequency_lowest_mhz
    elif args.frequency_mhz_begin < frequency_lowest_mhz:
            raise ValueError(f"Specified begin frequency is out of bounds: {frequency_lowest_mhz} Hz")

    if args.frequency_mhz_end is None:
        args.frequency_mhz_end = frequency_highest_mhz
    elif args.frequency_mhz_end > frequency_highest_mhz:
            raise ValueError(f"Specified end frequency is out of bounds: {frequency_highest_mhz} Hz")
    
    frequency_begin_fineidx = len(list(filter(lambda x: x<-upchan_bw, frequencies_mhz-args.frequency_mhz_begin)))
    frequency_end_fineidx = len(list(filter(lambda x: x<=0, frequencies_mhz-args.frequency_mhz_end)))
    assert frequency_begin_fineidx != frequency_end_fineidx, f"{frequency_begin_fineidx} == {frequency_end_fineidx}"

    pycorr.logger.info(f"Fine-frequency channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
    pycorr.logger.info(f"Fine-frequency range: [{frequencies_mhz[frequency_begin_fineidx]}, {frequencies_mhz[frequency_end_fineidx-1]}] MHz")

    frequency_begin_coarseidx = int(numpy.floor(frequency_begin_fineidx/args.upchannelisation_rate))
    frequency_end_coarseidx = int(numpy.ceil(frequency_end_fineidx/args.upchannelisation_rate))

    if frequency_end_coarseidx == frequency_begin_coarseidx:
        if frequency_end_coarseidx <= nchan:
            frequency_end_coarseidx += 1
        else:
            assert frequency_begin_coarseidx >= 1
            frequency_begin_coarseidx -= 1

    pycorr.logger.info(f"Coarse-frequency channel range: [{frequency_begin_coarseidx}, {frequency_end_coarseidx})")
    pycorr.logger.info(f"Coarse-frequency range: [{frequencies_mhz[frequency_begin_coarseidx*args.upchannelisation_rate]}, {frequencies_mhz[frequency_end_coarseidx*args.upchannelisation_rate - 1]}] MHz")
    assert frequency_begin_coarseidx != frequency_end_coarseidx
    
    frequencies_mhz = frequencies_mhz[frequency_begin_fineidx:frequency_end_fineidx]

    frequency_end_fineidx = frequency_end_fineidx - frequency_begin_coarseidx*args.upchannelisation_rate
    frequency_begin_fineidx = frequency_begin_fineidx - frequency_begin_coarseidx*args.upchannelisation_rate

    pycorr.logger.info(f"Fine-frequency relative channel range: [{frequency_begin_fineidx}, {frequency_end_fineidx})")
    pycorr.logger.info(f"Fine-frequency range: [{frequencies_mhz[0]}, {frequencies_mhz[-1]}] MHz")
    frequencies_mhz += 0.5 * upchan_bw

    guppi_header["POLS"] = guppi_header.get("POLS", args.polarisations)
    polarisation_chars = guppi_header["POLS"]
    assert len(polarisation_chars) == npol
    polproducts = [
        f"{pol1}{pol2}"
        for pol1 in polarisation_chars for pol2 in polarisation_chars
    ]

    phase_center_radians = (
        _degrees_process(guppi_header.get("RA_PHAS", guppi_header["RA_STR"])) * numpy.pi / 12.0 ,
        _degrees_process(guppi_header.get("DEC_PHAS", guppi_header["DEC_STR"])) * numpy.pi / 180.0 ,
    )
    
    timeperblk = guppi_data.shape[2]
    piperblk = guppi_header.get("PIPERBLK", timeperblk)
    tbin = guppi_header.get("TBIN", 1.0/coarse_chan_bw)
    synctime = guppi_header.get("SYNCTIME", 0)
    dut1 = guppi_header.get("DUT1", 0.0)

    time_array = numpy.array((num_bls,), dtype='d')
    integration_time = numpy.array((num_bls,))
    integration_time.fill(args.upchannelisation_rate*args.integration_rate*tbin)
    flags = numpy.zeros((num_bls, len(frequencies_mhz), len(polproducts)), dtype='?')
    nsamples = numpy.ones(flags.shape, dtype='d')

    ant_xyz = numpy.array([ant["position"] for ant in telinfo["antennas"]])
    antenna_numbers = [ant["number"] for ant in telinfo["antennas"]]
    baseline_ant_1_indices = [antenna_numbers.index(antnum) for antnum in ant_1_array]
    baseline_ant_2_indices = [antenna_numbers.index(antnum) for antnum in ant_2_array]
    lla = (telinfo["longitude"], telinfo["latitude"], telinfo["altitude"])

    with h5py.File(output_filepath, "w") as f:

        uvh5_datasets = pycorr.uvh5_initialise(
            f,
            telinfo["telescope_name"],
            guppi_header.get("TELESCOP", "UNKNOWN"), # instrument_name
            guppi_header.get("SRC_NAME", "UNKNOWN"), # instrument_name,
            lla,
            telinfo["antennas"],
            frequencies_mhz*1e6,
            polproducts,
            phase_center_radians,
            dut1 = dut1
        )

        datablock_shape = list(guppi_data.shape)
        datablocks_per_requirement = datablock_time_requirement/datablock_shape[2]
        pycorr.logger.debug(f"Collects ceil({datablocks_per_requirement}) blocks for correlation.")
        datablock_shape[2] = numpy.ceil(datablocks_per_requirement)*datablock_shape[2]
        
        datablock = guppi_data[:, :, 0:0, :]
        datablock_pktidx_start = guppi_header["PKTIDX"]

        t = time.time()
        t_start = t
        last_file_pos = 0
        datasize_processed = 0
        integration_count = 0
        # Integrate fine spectra in a separate buffer
        integration_buffer = numpy.zeros(flags.shape, dtype="D")
        while True:
            datablock = numpy.concatenate(
                (datablock, guppi_data),
                axis=2 # concatenate in time
            )
            
            if datablock.shape[2] >= datablock_time_requirement:
                file_pos = guppi.file.tell()
                datasize_processed += file_pos - last_file_pos
                last_file_pos = file_pos
                progress = datasize_processed/guppi_bytes_total
                elapsed_s = time.time() - t_start
                pycorr.logger.info(f"Progress: {datasize_processed/10**6:0.3f}/{guppi_bytes_total/10**6:0.3f} MB ({100*progress:03.02f}%). Elapsed: {elapsed_s:0.3f} s, ETC: {elapsed_s*(1-progress)/progress:0.3f} s")
                pycorr.logger.debug(f"Running throughput: {datasize_processed/(elapsed_s*10**6):0.3f} MB/s")

            while datablock.shape[2] >= datablock_time_requirement:
                datablock_residual = datablock[:, :, datablock_time_requirement:, :]
                datablock = datablock[:, :, 0:datablock_time_requirement, :]

                datablock_bytesize = datablock.size * datablock.itemsize

                t = time.time()
                datablock = pycorr.upchannelise(
                    datablock[:, frequency_begin_coarseidx:frequency_end_coarseidx, :, :],
                    args.upchannelisation_rate
                )[:, frequency_begin_fineidx:frequency_end_fineidx, :, :]
                
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Channelisation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                datablock_bytesize = datablock.size * datablock.itemsize

                t = time.time()
                datablock = pycorr.correlation(datablock)
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Correlation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                t = time.time()
                assert datablock.shape[2] == 1
                integration_buffer += datablock.reshape(integration_buffer.shape)
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Integration {integration_count}/{args.integration_rate}: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                integration_count += 1

                datablock = datablock_residual
                del datablock_residual

                if integration_count < args.integration_rate:
                    continue
                
                t = time.time()
                time_array.fill(
                    _get_jd(
                        tbin,
                        ntimes,
                        piperblk,
                        synctime,
                        datablock_pktidx_start + (datablock_time_requirement/2)*piperblk/timeperblk
                    )
                )

                pycorr.uvh5_write_chunk(
                    uvh5_datasets,
                    ant_1_array,
                    ant_2_array,
                    pycorr.get_uvw_array(
                        time_array[0],
                        phase_center_radians,
                        ant_xyz,
                        lla,
                        baseline_ant_1_indices,
                        baseline_ant_2_indices,
                        dut1=dut1,
                    ),
                    time_array,
                    integration_time,
                    integration_buffer,
                    flags,
                    nsamples,
                )
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Write: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                
                datablock_pktidx_start += datablock_time_requirement*piperblk/timeperblk
                
                integration_count = 0
                integration_buffer.fill(0.0)

            guppi_header, guppi_data = guppi.read_next_block()
            if guppi_header is None and len(args.raw_filepaths) > 0:
                guppi = Guppi(args.raw_filepaths.pop(0))
                last_file_pos = 0
                guppi_header, guppi_data = guppi.read_next_block()
                
            if guppi_header is None:
                break
