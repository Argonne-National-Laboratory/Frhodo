"""Path-settings controller — persists last-used directories + the ``Dir.ini`` file."""
import configparser
import os
import re
import shutil
import stat

import numpy as np
from qtpy import QtCore


class Path:
    """Tracks the experiment / mechanism / save directories across sessions.

    Reads ``Dir.ini`` at startup, writes it on directory changes.
    Exposes the resolved directories via ``parent.path[...]``.
    """

    def __init__(self, parent, path):
        self.parent = parent
        self.loading_dir_file = False

        parent.path = path
        parent.path["graphics"] = parent.path["package"] / "ui" / "graphics"
        self.config = configparser.RawConfigParser()

        # Specify yaml files
        parent.path["default_config"] = parent.path["appdata"] / "default_config.yaml"
        parent.path["Cantera_Mech"] = parent.path["appdata"] / "generated_mech.yaml"
        for key in ["default_config", "Cantera_Mech"]:
            if parent.path[key].exists():  # Check that file is readable and writable
                if not os.access(parent.path[key], os.R_OK) or not os.access(
                    parent.path[key], os.W_OK
                ):
                    os.chmod(
                        parent.path[key], stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP
                    )  # try to change if not

        # Create file watcher
        self.fs_watcher = QtCore.QFileSystemWatcher()
        self.fs_watcher.directoryChanged.connect(self.mech)

    def mech(self):
        parent = self.parent

        # Test for existance of mech folder and return if it doesn't
        if parent.path["mech_main"].is_dir():
            self.mech_main_exists = True
        else:
            self.mech_main_exists = False

        unsorted_mech_files = []
        thermo_files = []
        trans_files = []
        for file in parent.path["mech_main"].glob("*"):
            if not file.is_file():
                continue

            name = file.name
            ext = file.suffix.lower()

            if ext == ".therm":
                thermo_files.append(name)
            if ext == ".tran":  # transport files preserved for future use
                trans_files.append(name)
            elif ext in [
                ".yaml",
                ".yml",
                ".cti",
                ".ck",
                ".mech",
                ".inp",
            ]:  #  '.ctml', '.xml' intentionally unsupported
                if "generated_mech.yaml" == name:
                    continue
                elif "generated_mech.yml" == name:
                    continue
                unsorted_mech_files.append(name)

        # Sort Mechs
        mech = {"base": [], "suffix": [], "num": [], "ext": []}
        suffix = " - Opt "
        for file in unsorted_mech_files:
            match_opt = re.findall(r"(.*){:s}(\d+).(.*)".format(suffix), file)
            if (
                match_opt is not None and len(match_opt) > 0
            ):  # if match, then optimized mech with number
                mech["base"].append(match_opt[0][0])
                mech["suffix"].append(suffix)
                mech["num"].append(int(match_opt[0][1]))
                mech["ext"].append(match_opt[0][2])
            else:  # if no - Opt then it's a new mechanism
                match = re.findall(r"(.*)\.(.*)", file)
                mech["base"].append(match[0][0])
                mech["suffix"].append("")
                mech["num"].append(np.inf)  # Makes unedited mech first
                mech["ext"].append(match[0][1])

        # Sort by descending opt num and default sorting of mech name
        sort_idx = np.lexsort((-np.array(mech["num"]), mech["base"]))
        mech_files = []
        for n in sort_idx:
            if mech["suffix"][n] != "":
                num = str(mech["num"][n])
            else:
                num = ""
            name = mech["base"][n] + mech["suffix"][n] + num + "." + mech["ext"][n]
            mech_files.append(name)

        # Add items to combobox
        for obj in [parent.mech_select_comboBox, parent.thermo_select_comboBox]:
            obj.blockSignals(True)
            oldText = obj.currentText()
            obj.clear()
            if obj is parent.mech_select_comboBox:
                obj.addItems(mech_files)
            else:
                obj.addItems(thermo_files)

            idx = obj.findText(oldText)  # if the previous selection exists, reselect it
            obj.blockSignals(False)
            if idx >= 0:
                obj.setCurrentIndex(idx)

    def shock_paths(self, prefix, ext, max_depth=2):
        parent = self.parent
        path = parent.path
        path["shock"] = parent.path["exp_main"]

        shock_num = np.array([]).astype(int)
        shock_path = []
        for file in parent.path["exp_main"].rglob("*"):
            if not file.is_file():  # only looking for files
                continue

            depth = len(file.relative_to(parent.path["exp_main"]).parts)
            if depth > max_depth:  # skip if the depth is greater than allowed
                continue

            match = re.search(
                rf"{prefix}.*\.{ext}$", file.name
            )
            if match:
                numMatch = re.search(r"(\d+)", file.name)  # search for numbers
                n = int(numMatch.group(0))

                # skip appending duplicates with larger path depth
                append_shock = True
                shock_match = np.argwhere(shock_num == n)
                if np.shape(shock_match)[0] > 0:
                    i = shock_match[0, 0]
                    depth_old = len(
                        shock_path[i].relative_to(parent.path["exp_main"]).parts
                    )
                    if depth > depth_old:
                        append_shock = False

                if append_shock:
                    shock_num = np.append(shock_num, n)
                    shock_path.append(
                        file.parents[0]
                    )  # appends root directory of shock

        if len(shock_num) == 0:  # in case no shocks in directory
            return []

        # Sort based on shock number
        idx_sort = np.argsort(shock_num)
        shock_num = shock_num[idx_sort]
        shock_path = np.array(shock_path)[idx_sort]

        # Create sorted list of shock_num and shock_path
        return np.column_stack((shock_num, shock_path))

    def shock(self, shock_num):
        def find_nearest(array, value):  # Finds the nearest value
            array = np.asarray(array)
            if value in array:
                idx = np.where(value == array)[0][0]
            elif np.max(array) < value:
                idx = np.argmax(array)
            elif np.min(array) > value:
                idx = np.argmin(array)
            else:
                idx = np.argmin(np.abs(array - value))

            return idx, array[idx]

        parent = self.parent

        sel = parent.shock_selection
        shock_num_arr = np.asarray(shock_num)
        change = sel.current - sel.previous
        if np.abs(change) == 1:  # if stepping through experiments
            idx = np.where(shock_num_arr == sel.previous)[0] + change

            if np.shape(idx)[0] == 0:  # if shock cannot be found, find nearest
                idx, Shock_Choice = find_nearest(shock_num_arr, sel.current)
            else:  # if shock can be found, step
                idx = idx[0]
                if idx >= len(shock_num_arr):
                    idx = 0
                elif idx < 0:
                    idx = len(shock_num_arr) - 1
                Shock_Choice = shock_num_arr[idx]
        else:  # if selecting experiment or loading a new exp directory
            idx, Shock_Choice = find_nearest(shock_num_arr, sel.current)

        sel.current = Shock_Choice
        for box in parent.shock_choice_box.twin:
            box.blockSignals(True)
            box.setValue(Shock_Choice)
            box.blockSignals(False)

        return idx

    def shock_output(self):
        parent = self.parent
        log = parent.log

        # Add Exp_set_name if exists
        shock_num = str(parent.shock_selection.current)
        if not parent.display_shock.series_name:
            parent.path["output_dir"] = parent.path["sim_main"] / (
                "Shock " + str(shock_num)
            )
        else:
            parent.path["output_dir"] = (
                parent.path["sim_main"] / parent.display_shock.series_name
            ) / ("Shock " + str(shock_num))

        # Create folders if needed
        if not parent.path["output_dir"].exists():
            try:
                parent.path["output_dir"].mkdir(exist_ok=True, parents=True)
            except (IOError, FileNotFoundError) as e:
                log.append("Error in saving:")
                log.append(e)
                return

        parent.path["Sim log"] = parent.path["output_dir"] / "Sim log.txt"

        # Find next sim number based on Sim log
        self.sim_num = 0
        if parent.path["Sim log"].exists():
            with open(parent.path["Sim log"], "r") as f:
                for line in f:
                    if len(re.findall(r"Sim \d+:", line)) > 0:
                        self.sim_num += 1

        self.sim_num += 1

        # Assign sim directory
        if self.sim_num == 1:
            parent.path["sim_dir"] = parent.path["output_dir"]
        elif self.sim_num > 1:
            parent.path["sim_dir"] = parent.path["output_dir"] / "Sim {:d}".format(
                self.sim_num
            )
            parent.path["sim_dir"].mkdir(exist_ok=True, parents=True)

            # Move files if second sim
            if self.sim_num == 2:
                sim_1_dir = parent.path["output_dir"] / "Sim 1"
                sim_1_dir.mkdir(exist_ok=True, parents=True)
                for entry in parent.path["output_dir"].glob("*"):
                    if entry.is_file():
                        if (
                            len(re.findall(r"Sim \d+ - ", entry.name)) > 0
                        ):  # if files starts with Sim ####
                            shutil.move(entry, sim_1_dir / entry.name)

        for file in ["Mech.yaml", "Mech.ck", "Plot.png", "Legend.txt"]:
            parent.path[file] = parent.path["sim_dir"] / "Sim {:d} - {:s}".format(
                self.sim_num, file
            )

    def sim_output(self, var_name):  # takes variable name and creates path for it
        if (
            var_name == "± % |Density Gradient|"
        ):  # lots of invalid characters, replace
            var_name = "signed % Abs Density Gradient"

        name = "Sim {:d} - {:s}.txt".format(self.sim_num, var_name)
        self.parent.path[var_name] = self.parent.path["sim_dir"] / name

        return self.parent.path[var_name]

    def optimized_mech(self, file_out="opt_mech"):
        parent = self.parent

        mech_name = parent.path["mech"].stem
        mech_name = re.sub(
            r" - Opt \d+$", "", str(mech_name)
        )  # strip opt and trailing number
        mech_name += " - Opt "  # add opt back in

        num = [0]
        for file in parent.path["mech_main"].glob("*"):
            if not file.is_file():
                continue

            num_found = re.findall(
                r"{:s}\s*(-?\d+(?:\.\d+)?)".format(mech_name), file.name
            )
            if len(num_found) > 0:
                num.append(*[int(num) for num in num_found])

        opt_mech_file = "{:s}{:.0f}.mech".format(mech_name, np.max(num) + 1)
        recast_mech_file = opt_mech_file.replace("Opt", "PreOpt")
        parent.path["Optimized_Mech.mech"] = parent.path["mech_main"] / opt_mech_file
        parent.path["Optimized_Mech_recast.mech"] = (
            parent.path["mech_main"] / recast_mech_file
        )

        if file_out == "opt_mech":
            return parent.path["Optimized_Mech.mech"]
        elif file_out == "recast_mech":
            return parent.path["Optimized_Mech_recast.mech"]

    def load_dir_file(self, file_path):
        parent = self.parent
        self.loading_dir_file = True
        self.config.read(file_path)

        # loading exp_main creates a new series
        parent.exp_main_box.setPlainText(self.config["Directories"]["exp_main"])

        if (
            "exp_main" not in parent.directory.invalid
            and ": " in self.config["Species Default Aliases"]["aliases"]
        ):
            for pair in self.config["Species Default Aliases"]["aliases"].split("; "):
                exp_name, thermo_name = pair.split(": ")
                parent.series.current["species_alias"][exp_name] = thermo_name

        parent.mech_main_box.setPlainText(self.config["Directories"]["mech_main"])
        parent.sim_main_box.setPlainText(self.config["Directories"]["sim_main"])
        if len(self.config["Experiment Set Name"]["name"]) > 0:
            parent.exp_series_name_box.setText(
                self.config["Experiment Set Name"]["name"]
            )

        self.mech()  # This updates the mech and thermo combo boxes
        self.loading_dir_file = False

    def save_dir_file(self, file_path):
        self.config["Experiment Set Name"] = {
            "name": self.parent.display_shock.series_name
        }

        self.config["Species Default Aliases"] = {"aliases": self._alias_str()}

        self.config["Directories"] = {
            "exp_main": self.parent.path["exp_main"],
            "mech_main": self.parent.path["mech_main"],
            "sim_main": self.parent.path["sim_main"],
        }

        with open(file_path, "w") as configfile:
            self.config.write(configfile)

    def save_aliases(self, file_path):
        self.config.set("Species Default Aliases", "aliases", self._alias_str())

        with open(file_path, "w") as configfile:
            self.config.write(configfile)

    def _alias_str(self):
        species_aliases_str = []
        for alias, species in self.parent.series.current["species_alias"].items():
            species_aliases_str.append(alias + ": " + species)

        return "; ".join(species_aliases_str)

    def set_watch_dir(self):
        if self.fs_watcher.directories():
            self.fs_watcher.removePaths(self.fs_watcher.directories())

        if self.parent.path["mech_main"].is_dir():
            self.fs_watcher.addPath(str(self.parent.path["mech_main"]))
