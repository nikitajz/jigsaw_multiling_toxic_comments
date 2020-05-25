import datetime
import logging
import os
import sys
from pprint import pformat

from transformers import HfArgumentParser


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round(elapsed))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


def load_or_parse_args(ArgsSeq, verbose=False):
    """Load arguments from json if only one parameter with .json extension is provided. 
    Otherwise parse arguments provided in command line.

    Parameters
    ----------
    ArgsSeq : Iterable[Dataclass]
        Sequence of dataclass, e.g. (ModelArgs, TrainingArgs)
    verbose : bool
        Whether to print parsed arguments

    Returns
    -------
    Iterable[Dataclass]
        Return parsed args one per each provided dataclass
    """
    parser = HfArgumentParser(ArgsSeq)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        seq_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        seq_args = parser.parse_args_into_dataclasses()

    if verbose:
        logger = logging.getLogger(__name__)
        for arg_pack in seq_args:
            logger.info(f'{arg_pack.__class__.__name__}:\n{pformat(arg_pack.__dict__)}\n')

    return seq_args
