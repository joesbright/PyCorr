import os, glob, argparse, yaml, logging, time

import tomli as tomllib # `tomllib` as of Python 3.11 (PEP 680)
import h5py
import numpy
from guppi.guppi import Guppi

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

def _get_telescope_metadata(telescope_info_filepath):
    """
    Returns a standardised formation of the TOML/YAML contents:
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
    else:
        raise ValueError(f"Unknown file format '{telinfo_ext}' ({os.path.basename(telescope_info_filepath)}). Known formats are: yaml, toml.")

    longitude = _degrees_process(telescope_info["longitude"])
    latitude = _degrees_process(telescope_info["latitude"])
    altitude = telescope_info["altitude"]
    antenna_positions = numpy.array([antenna["position"] for antenna in telescope_info["antennas"]])

    if "ecef" == telescope_info["antenna_position_frame"].lower():
        logger.info("Transforming antenna positions from XYZ to ECEF")
        transform_antenna_positions_ecef_to_xyz(
            longitude,
            latitude,
            altitude,
            antenna_positions,
        )
    else:
        # TODO handle enu
        assert telescope_info["antenna_position_frame"].lower() == "xyz"

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
    for i in range(100):
        key = f"ANTNMS{i:02d}"
        if key in guppi_header:
            if guppi_header.get("TELESCOP", "UNKNOWN") == "ATA":
                # ATA antenna names have LO suffixed
                antenna_names += map(lambda name: name[:-1], guppi_header[key].split(","))
            else:
                antenna_names += guppi_header[key].split(",")

    nants = guppi_header.get("NANTS", 1)
    antenna_names = antenna_names[:nants]
    antenna_telinfo = {
        antenna["name"]: antenna
        for antenna in telinfo["antennas"]
        if antenna["name"] in antenna_names
    }
    assert len(antenna_telinfo) == len(antenna_names), f"Telescope information does not cover RAW listed antenna: {set(antenna_names).difference(set([ant['name'] for ant in telinfo]))}"
    
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
        help="The path to telescope information.",
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
        help="Increase the verbosity of the generation (0=Error, 1=Warn, 2=Info, 3=Debug)."
    )

    args = parser.parse_args()
    pycorr.logger.setLevel(
        [
            logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
        ][args.verbose]
    )

    datablock_time_requirement = args.upchannelisation_rate * args.integration_rate

    telinfo = _get_telescope_metadata(args.telescope_info_filepath)
    if len(args.raw_filepaths) == 1 and not os.path.exists(args.raw_filepaths[0]):
        pycorr.logger.info(f"Given RAW filepath does not exist, assuming it is the stem.")
        args.raw_filepaths = glob.glob(f"{args.raw_filepaths[0]}*.raw")
        pycorr.logger.info(f"Found {args.raw_filepaths}.")
    
    input_dir, input_filename = os.path.split(args.raw_filepaths[0])
    if args.output_filepath is None:
        output_filepath = os.path.join(input_dir, f"{os.path.splitext(input_filename)[0]}.bfr5")
    else:
        output_filepath = args.output_filepath

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
    frequency_channel_0_hz = guppi_header["OBSFREQ"] - (nchan/2 + schan - 0.5)*guppi_header["CHAN_BW"]
    upchan_bw = guppi_header["CHAN_BW"]/args.upchannelisation_rate
    frequency_channel_0_hz += 0.5 * upchan_bw
    frequencies_hz = (frequency_channel_0_hz + numpy.arange(nchan*args.upchannelisation_rate)*upchan_bw)*1e6

    guppi_header["POLS"] = guppi_header.get("POLS", args.polarisations)
    polarisation_chars = guppi_header["POLS"]
    assert len(polarisation_chars) == npol
    polproducts = [
        f"{pol1}{pol2}"
        for pol1 in polarisation_chars for pol2 in polarisation_chars
    ]

    phase_center_radians = (
        float(guppi_header.get("RA_PHAS", guppi_header["RA_STR"])) * numpy.pi / 12.0 ,
        float(guppi_header.get("DEC_PHAS", guppi_header["DEC_STR"])) * numpy.pi / 180.0 ,
    )
    
    timeperblk = guppi_data.shape[2]
    piperblk = guppi_header.get("PIPERBLK", timeperblk)
    tbin = guppi_header.get("TBIN", 1.0/guppi_header["CHAN_BW"])
    synctime = guppi_header.get("SYNCTIME", 0)
    dut1 = guppi_header.get("DUT1", 0.0)

    time_array = numpy.array((num_bls,), dtype='d')
    integration_time = numpy.array((num_bls,))
    integration_time.fill(datablock_time_requirement*tbin)
    flags = numpy.zeros((num_bls, len(frequencies_hz), len(polproducts)), dtype='?')
    nsamples = numpy.ones((num_bls, len(frequencies_hz), len(polproducts)), dtype='d')

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
            frequencies_hz,
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
                pycorr.logger.debug(f"Datablock bytesize: {datablock_bytesize/(10**6)} MB")

                t = time.time()
                datablock = pycorr.upchannelise(datablock, args.upchannelisation_rate)
                
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Channelisation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                t = time.time()
                assert datablock.shape[2] == args.integration_rate
                datablock = pycorr.integrate(datablock)
                
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Integration: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

                t = time.time()
                corr = pycorr.correlation(datablock)
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Correlation: {datablock_bytesize/(elapsed_s*10**6)} MB/s")

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
                    corr,
                    flags,
                    nsamples,
                )
                elapsed_s = time.time() - t
                pycorr.logger.debug(f"Write: {datablock_bytesize/(elapsed_s*10**6)} MB/s")
                
                datablock_pktidx_start += datablock_time_requirement*piperblk/timeperblk
                datablock = datablock_residual
                del datablock_residual

            guppi_header, guppi_data = guppi.read_next_block()
            if guppi_header is None and len(args.raw_filepaths) > 0:
                guppi = Guppi(args.raw_filepaths.pop(0))
                last_file_pos = 0
                guppi_header, guppi_data = guppi.read_next_block()
                
            if guppi_header is None:
                break