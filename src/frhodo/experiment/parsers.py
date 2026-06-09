"""Tranter shock-experiment file parsers.

Loads ``.exp`` (Tranter v0/v1 INI-style metadata), ``.rho`` (CSV density
trace), and ``.sig`` (raw oscilloscope signal) files for a single shock.

Pure parsing; no Qt. Parser errors emit through ``logging.getLogger``;
the GUI's ``GuiLogHandler`` routes them to the log widget.
"""
import configparser
import csv
import logging
import re
import sys
from copy import deepcopy
from typing import Callable

import numpy as np
from dateutil.parser import parse


log = logging.getLogger(__name__)


class ExperimentLoader:
    """Loads a single shock's ``.exp`` / ``.rho`` / ``.sig`` files.

    ``convert_units(value, from_unit, to_unit_label)`` converts the raw
    Tranter values (T1 in °C, P1 in torr, u1 in mm/μs, P4 in psi) to
    Cantera SI units.
    """

    def __init__(self, convert_units: Callable):
        self.convert_units = convert_units
        self.load_style = "tranter_v1_0"

    def parameters(self, file_path):
        """Read shock parameters from a ``.exp`` file and convert to SI.

        Auto-detects the v0 vs v1 Tranter format from the first line.
        Tranter writes T1 in °C, P1 in torr, u1 in mm/μs, P4 in psi;
        all four are converted to Cantera SI before return.

        Returns:
            Dict carrying ``T1`` [K], ``P1`` [Pa], ``u1`` [m/s], ``P4``
            [Pa], ``Sample_Rate`` [MHz], ``exp_mix``, ``thermo_mix``.

        Raises:
            Exception: When the file is missing a critical field (T1,
                P1, SampRate, or the derived u1 inputs).
        """
        with open(file_path) as f:
            lines = f.read().splitlines()

        if lines[0] == "[Date]":
            self.load_style = "tranter_v1_0"
            parameters = self.read_tranter_exp_v1(lines)
        elif lines[0] == '"[Expt Parameters]"':
            self.load_style = "tranter_v0_1"
            parameters = self.read_tranter_exp_v0(lines)
        else:
            self.load_style = "tranter_v1_0"
            parameters = self.read_tranter_exp_v1(lines)

        # Tranter raw units: T1 [°C], P1 [torr], u1 [mm/μs], P4 [psi]
        parameters["T1"] = self.convert_units(parameters["T1"], "°C", "2ct")
        parameters["P1"] = self.convert_units(parameters["P1"], "torr", "2ct")
        parameters["u1"] = self.convert_units(parameters["u1"], "mm/μs", "2ct")
        parameters["P4"] = self.convert_units(parameters["P4"], "psi", "2ct")
        parameters["Sample_Rate"] *= 1e-6  # Hz -> MHz

        return parameters

    def read_tranter_exp_v1(self, lines):
        def get_config(section, key):
            val = self.config[section][key]
            for delimiter in ['"', "'"]:
                val = val.strip(delimiter)

            return val

        self.config = configparser.RawConfigParser()
        self.config.read_string("\n".join(lines))

        mix = {}
        for key in [item[0] for item in self.config.items("Mixture")]:
            match_opt = re.findall(r"mol_(\d+)_formula", key)
            if match_opt is not None and len(match_opt) > 0:
                species_num = match_opt[0]

                if not self.config.has_option(
                    "Mixture", "Mol_" + species_num + "_Mol frc"
                ):
                    continue

                species = get_config("Mixture", "Mol_" + species_num + "_Formula")
                mol_frac = float(
                    get_config("Mixture", "Mol_" + species_num + "_Mol frc")
                )
                if mol_frac != 0:
                    mix[species] = mol_frac

        if float(get_config("Expt Params", "P1")) == 0:
            raise Exception("Exception in Experiment File: P1 is zero")
        elif float(get_config("Expt Params", "T1")) == 0:
            raise Exception("Exception in Experiment File: T1 is zero")
        elif float(get_config("Expt Params", "SampRate")) == 0:
            raise Exception("Exception in Experiment File: Sample Rate is zero")

        if self.config.has_option("Expt Params", "VelatObs"):
            u1 = float(get_config("Expt Params", "VelatObs"))
            if u1 == 0:
                raise Exception(
                    "Exception in Experiment File: Velocity at observation is zero"
                )
        else:
            tOpt = float(get_config("Expt Params", "tOpt"))
            PT_spacing = float(get_config("Expt Params", "PT Spacing"))
            u1 = PT_spacing / tOpt
            if tOpt == 0:
                raise Exception("Exception in Experiment File: tOpt is zero")
            elif PT_spacing == 0:
                raise Exception("Exception in Experiment File: PT Spacing is zero")

        parameters = {
            "T1": float(get_config("Expt Params", "T1")),
            "P1": float(get_config("Expt Params", "P1")),
            "u1": u1,
            "P4": float(get_config("Expt Params", "P4")),
            "exp_mix": deepcopy(mix),
            "thermo_mix": deepcopy(mix),
            "Sample_Rate": float(get_config("Expt Params", "SampRate")),
        }

        return parameters

    def read_tranter_exp_v0(self, lines):
        parameters = {
            "T1": None,
            "P1": None,
            "u1": None,
            "exp_mix": {},
            "thermo_mix": {},
        }

        key = None
        processed = {"exp_mix": [], "shock_conditions": []}
        for line in lines:
            if line == '"[Thermochemistry]"':
                key = "exp_mix"
                continue
            elif line == '"[Start Conditions]"' or line == '"[Expt Times]"':
                key = "shock_conditions"
                continue
            elif line.isspace() or len(line) == 0:
                key = None
                continue

            if key is not None:
                processed[key].append(line)

        parameters["P1"] = float(processed["shock_conditions"][1])
        parameters["T1"] = float(processed["shock_conditions"][2])
        parameters["u1"] = 120.0 / float(
            processed["shock_conditions"][-1]
        )  # assumes 120 mm spacing
        parameters["P4"] = 1.0  # placeholder; not used by current loaders

        for line in processed["exp_mix"]:
            species, mol_frac = line[1:].split(";")[:2]
            parameters["exp_mix"][species] = float(mol_frac)

        parameters["thermo_mix"] = parameters["exp_mix"]
        parameters["Sample_Rate"] = 50000000.0

        return parameters

    def csv(self, file):
        """Read a CSV into a numeric array plus a list of non-numeric rows.

        Returns:
            ``(data, nonnumeric)``. ``data`` is an ``np.ndarray`` of
            float values; ``nonnumeric`` is a list of ``[row_idx,
            first_cell]`` for rows that weren't parseable as numbers
            (typically headers). ``None`` for both when ``file`` is
            ``None``.

        Raises:
            Exception: Non-finite values in the data array, or no
                numeric rows at all.
        """
        def is_numeric(strings):
            for str in strings:
                try:
                    float(str)
                except ValueError:
                    return False

            return True

        if file is None:
            return None

        data = []
        data_nonnumeric = []
        with open(file, newline="") as f:
            reader = csv.reader(f)
            try:
                for n, row in enumerate(reader):
                    if is_numeric(row):
                        data.append(row)
                    else:
                        data_nonnumeric.append([n, row[0]])
            except csv.Error as e:
                sys.exit("file {}, line {}: {}".format(file.name, reader.line_num, e))

        data = np.array(data, float)

        if np.logical_not(np.isfinite(data)).any():
            raise Exception(
                "Exception in {:s}: Nonfinite values found in data".format(file.name)
            )
        elif np.shape(data)[0] == 0:
            raise Exception("Exception in {:s}: No data found".format(file.name))

        return data, data_nonnumeric

    def exp_data(self, file_path):
        exp_data, nonnumeric = self.csv(file_path)

        if self.load_style == "tranter_v0_1":
            exp_data = np.array(exp_data)
            exp_data = exp_data[:, [0, 2]]
            exp_data = exp_data[:-1, :]

        return exp_data

    def raw_signal(self, file_path):
        def is_date(string, fuzzy=False):
            try:
                parse(string, fuzzy=fuzzy)
                return True

            except ValueError:
                return False

        raw_sig, nonnumeric = self.csv(file_path)
        if nonnumeric:
            if is_date(nonnumeric[0][1].replace("-", " "), fuzzy=True):
                raw_sig = raw_sig[2:]

        return raw_sig

    def load_data(self, shock_num, main_path):
        """Locate and load every file belonging to one shock number.

        Globs ``main_path`` for files named ``Shock<N>.exp``,
        ``Shock<N>.rho``, and ``Shock<N>raw1.sig`` (plus the legacy
        LS L-R alias for the raw signal) and dispatches each to the
        matching parser.

        Returns:
            Dict keyed by ``"Shock.exp"``, ``"Shock.rho"``, and
            ``"ShockRaw1.sig"``. Missing files surface as ``None``
            with a logged warning rather than raising.
        """
        def try_load(fcn, path):
            try:
                return fcn(path)
            except Exception as e:
                log.warning("Error loading Shock %d: %s", shock_num, e)

        shock_num = int(shock_num)
        paths = {"Shock.exp": None, "Shock.rho": None, "ShockRaw1.sig": None}
        for item in main_path.glob("*"):
            if item.is_file():
                if "Shock{:d}.exp".format(shock_num) in item.name:
                    paths["Shock.exp"] = item
                elif "Shock{:d}.rho".format(shock_num) in item.name:
                    paths["Shock.rho"] = item
                elif "Shock{:d}raw1.sig".format(shock_num) in item.name:
                    paths["ShockRaw1.sig"] = item
                elif all(x in item.name for x in ["LS", "L-R"]):
                    paths["ShockRaw1.sig"] = item

        data = {}
        load_fcn = {
            "Shock.exp": self.parameters,
            "Shock.rho": self.exp_data,
            "ShockRaw1.sig": self.raw_signal,
        }
        for file, path in paths.items():
            if path is None:
                log.warning("Shock %d: %s is missing", shock_num, file)
                data[file] = None
            else:
                data[file] = try_load(load_fcn[file], path)
                if file == "Shock.rho" and data[file] is not None:
                    data[file][:, 0] *= 1e-6

            if data[file] is None:
                if file == "Shock.exp":
                    data[file] = {}
                else:
                    data[file] = np.array([])

        return data.values()
