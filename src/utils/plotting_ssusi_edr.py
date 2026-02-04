import os
import re
import glob
from dotenv import load_dotenv
from utils.ssusi_edr import EDRPass
from utils.time_dict import TimeDict
import datetime
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geospacepy.satplottools import draw_dialplot
from matplotlib.gridspec import GridSpec
from sklearn.neighbors import NearestNeighbors
import matplotlib.colors as clrs
import scipy.ndimage as im_tools
from geospacepy.special_datetime import (
    doyarr2datetime,
    datetimearr2jd,
    datetime2jd,
    jd2datetime,
    datetime2doy,
    jdarr2datetime,
)
from spacepy import pycdf
import time

load_dotenv()


class SUSSI_Plotter:
    def __init__(self, drive_path):
        self.edr_files = []
        self.hemishphere = "S"
        self.radiance_type = "LBHS"
        self.drive_path = drive_path
        self.ae_index = None
        self.meta_data_csv = pd.DataFrame(columns=["Filename", "Datetime", "Satellite", "Hemisphere", "Orbit", "AE_Index", "EDR_File"])

    def build_ae_index(self, ae_df):
        # Switch to hour by hour
        """Build a dictionary mapping dates to AE index values."""
        ae_index = {}
        for _, row in ae_df.iterrows():
            epoch = row['Epoch']
            ae_index[epoch] = row["AE"]
        self.ae_index = TimeDict(ae_index)
        return self.ae_index

    def yday_from_date(self, date):
        """Return integer of form YYYYDOY given a datetime.date."""
        year = date.year
        doy = date.timetuple().tm_yday
        return int(f"{year}{doy}")

    def date_to_yday_folder(self, date):
        """Return YYYYDOY string from datetime.date."""
        return f"{date.year}{date.timetuple().tm_yday}"

    def find_midnight_passes(self, root_dir, start_date=None, end_date=None):
        """
        Find dates, satellites, and hemispheres with SSJ measurements near midnight MLT (±1 hr).

        Parameters
        ----------
        root_dir : str
            Root directory containing YYYYDOY subfolders.
        start_date : datetime.date, optional
            Earliest date to include.
        end_date : datetime.date, optional
            Latest date to include.

        Returns
        -------
        list of (datetime.date, str, str)
            List of (date, satellite, hemisphere) triples.
        """
        midnight_passes = set()

        # convert start/end to YYYYDOY integers for comparison
        start_yday = self.yday_from_date(start_date) if start_date else None
        end_yday = self.yday_from_date(end_date) if end_date else None
        for subdir in sorted(os.listdir(root_dir)):
            if not subdir.isdigit():
                continue  # skip non-date folders

            try:
                subdir_val = int(subdir)
            except ValueError:
                continue

            # apply date filtering
            if start_yday and subdir_val < start_yday:
                continue
            if end_yday and subdir_val > end_yday:
                continue
            folder_path = os.path.join(root_dir, subdir)
            if not os.path.isdir(folder_path):
                continue

            # loop through .cdf files in this subfolder
            for fname in os.listdir(folder_path):
                if not fname.endswith(".cdf"):
                    continue

                filepath = os.path.join(folder_path, fname)

                # extract satellite (e.g. "f16") from filename
                sat_match = re.search(r"f\d{2}", fname.lower())
                satellite = sat_match.group(0).upper() if sat_match else "UNKNOWN"

                try:
                    with pycdf.CDF(filepath) as cdffile:
                        epoch = cdffile["Epoch"][:].flatten()

                        try:
                            mlt = cdffile["SC_APEX_MLT"][:].flatten()
                            lat = cdffile["SC_APEX_LAT"][:].flatten()
                        except:
                            mlt = cdffile["SC_AACGM_LTIME"][:].flatten()
                            lat = cdffile["SC_AACGM_LAT"][:].flatten()

                        datetimes = epoch  # already datetime objects

                        for t, m, la in zip(datetimes, mlt, lat):
                            if m >= 23 or m <= 1:
                                hemisphere = "N" if la >= 0 else "S"
                                midnight_passes.add((t.date(), satellite, hemisphere))
                                break  # move to next file once found
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")

        # sort by date, then satellite, then hemisphere
        return sorted(midnight_passes, key=lambda x: (x[0], x[1], x[2]))

    def plot_erd_file(self, edr_file, show_plot_info, min_ae):
        hemisphere = "S"  # N or S
        radiance_type = "LBHS"
        orb_match = re.search(r"REV(\d{5})", edr_file)
        if orb_match:
            orbit_num = int(orb_match.group(1))  # get orbit number
        else:
            print("Orbit number not found.")
        sat_match = re.search(r"dmspf(\d{2})", edr_file)
        if sat_match:
            dmsp = int(sat_match.group(1))  # get satellite number
        else:
            print("Sat number not found.")
        # create pass object containing data for this particular image/pass
        ssusi_obs = EDRPass(
            edr_file,
            dmsp,
            hemisphere,
            radiance_type=radiance_type,
            noise_removal=False,
            spatial_bin=False,
        )
        print("Pass Start Time:", jd2datetime(np.nanmin(ssusi_obs["jds"])))
        print("Pass End Time:", jd2datetime(np.nanmax(ssusi_obs["jds"])))

        startdt = jd2datetime(np.nanmin(ssusi_obs["jds"]))
        enddt = jd2datetime(np.nanmax(ssusi_obs["jds"]))

        ae_time, ae_val = self.ae_index.closest(enddt)
        if ae_val < min_ae:
            return

        dt = datetime.datetime(startdt.year, startdt.month, startdt.day)
        date = datetime.date(startdt.year, startdt.month, startdt.day)
        f = plt.figure(figsize=(10, 10), dpi=150)
        ax_1 = f.add_subplot(111, projection="polar")
        ssusi_obs.plot_obs(ax_1, ptsize=1)  # use plotting function to plot
        # Remove legend
        for extra_ax in f.axes:
            if extra_ax != ax_1:
                f.delaxes(extra_ax)

        print(f"Closet time to {enddt} is {ae_time}")
        print(f"AE Index at {ae_time} is {ae_val}")

        if show_plot_info:
            title = f"DMSP {dmsp} {hemisphere} {radiance_type} {dt.month}-{dt.day}-{dt.year} Orbit {orbit_num} Daily AE Index {ae_val}"
            ax_1.set_title(
                title,
                fontsize=14,
            )

        file_name = f"{dmsp}_{hemisphere}_{radiance_type}_{enddt.strftime('%Y%m%d%H%M%S')}_{orbit_num}"
        meta_data_slice = {
            "Filename": file_name,
            "Datetime": enddt,
            "Satellite": dmsp,
            "Hemisphere": hemisphere,
            "Orbit": orbit_num,
            "AE_Index": ae_val,
            "EDR_File": os.path.basename(edr_file)
        }
        self.meta_data_csv = pd.concat([self.meta_data_csv, pd.DataFrame([meta_data_slice])], ignore_index=True)
        date_str = date.strftime("%Y-%m-%d")
        os.makedirs(os.path.join(self.drive_path, "Figures", date_str), exist_ok=True)
        file_name = os.path.join(self.drive_path, "Figures", date_str, f"{file_name}.png")
        if not show_plot_info:
            leg = ax_1.get_legend()
            if leg is not None:
                leg.remove()
        plt.savefig(file_name)
        if not show_plot_info:
            plt.close(f)

    def find_ssj_midnight_times(self, ssj_root, date, sat, hemi):
        """Find exact SSJ times near midnight for a given date/satellite/hemisphere."""
        folder = os.path.join(ssj_root, self.date_to_yday_folder(date))
        times = []

        if not os.path.isdir(folder):
            return times

        for fname in os.listdir(folder):
            if not fname.endswith(".cdf"):
                continue
            if sat.lower() not in fname.lower():  # only matching satellite
                continue

            filepath = os.path.join(folder, fname)
            try:
                with pycdf.CDF(filepath) as cdffile:
                    epoch = cdffile['Epoch'][:].flatten()
                    try:
                        mlt = cdffile['SC_APEX_MLT'][:].flatten()
                        lat = cdffile['SC_APEX_LAT'][:].flatten()
                    except:
                        mlt = cdffile['SC_AACGM_LTIME'][:].flatten()
                        lat = cdffile['SC_AACGM_LAT'][:].flatten()

                    datetimes = epoch  # already datetime objects
                    for t, m, la in zip(datetimes, mlt, lat):
                        if (m >= 23 or m <= 1):
                            h = "N" if la >= 0 else "S"
                            if h == hemi:
                                times.append(t)
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        return sorted(times)

    def parse_ssusi_filename(self, fname):
        """
        Parse SSUSI filename times.
        Example: dmspf16_ssusi_edr-aurora_2010001T204501-2010001T222653-REV32026_vA8.2.0r000.nc
        Returns: satellite (e.g. F16), start datetime, end datetime
        """
        sat_match = re.search(r"f\d{2}", fname.lower())
        sat = sat_match.group(0).upper() if sat_match else "UNKNOWN"

        match = re.search(r"_(\d{7}T\d{6})-(\d{7}T\d{6})", fname)
        if not match:
            return sat, None, None

        def parse_time(s):
            year = int(s[:4])
            doy = int(s[4:7])
            time = s[8:]
            dt = datetime.datetime.strptime(f"{year} {doy} {time}", "%Y %j %H%M%S")
            return dt

        start = parse_time(match.group(1))
        end = parse_time(match.group(2))
        return sat, start, end

    def match_ssusi_files(self, ssusi_root, date, sat, times):
        """Return SSUSI filenames covering the given times for date/satellite."""
        folder = os.path.join(ssusi_root, self.date_to_yday_folder(date))
        matches = []

        if not os.path.isdir(folder):
            return matches

        for fname in os.listdir(folder):
            if not fname.endswith(".nc"):
                continue
            if sat.lower() not in fname.lower():
                continue

            sat_name, start, end = self.parse_ssusi_filename(fname)
            if start is None or end is None:
                continue

            for t in times:
                if start <= t <= end:
                    matches.append(fname)
                    break
        return sorted(set(matches))

    def plot_midnight_passes_from_dates(self, dates, show_plot_info=True, min_ae=0):
        print("Finding midnight passes...")

        ssj_root = os.path.join(self.drive_path, "DMSP", "SSJ")
        ssusi_root = os.path.join(self.drive_path, "DMSP", "SSUSIEDR")
        for i, date in enumerate(dates): # O(n^3) however the number of passes per date should be small, and the number of matches per pass should also be small
            print(f"Processing date: {date}")
            print("---------------------")
            print(f"Date {i+1}/{len(dates)}")
            print("---------------------")
            midnight_passes = self.find_midnight_passes(ssj_root, date, date)
            print(f"Found {len(midnight_passes)} midnight passes")
            for j, mp in enumerate(midnight_passes):
                print(f"Processing pass {j+1}/{len(midnight_passes)}")
                date, sat, hemi = mp # Unpackes the tuples from the midnight pass
                times = self.find_ssj_midnight_times(ssj_root, date, sat, hemi)
                matches = self.match_ssusi_files(ssusi_root, date, sat, times)
                for m in matches:
                    edr_file = os.path.join(ssusi_root, self.date_to_yday_folder(date), m)
                    try:
                        self.plot_erd_file(edr_file=edr_file, show_plot_info=show_plot_info, min_ae=min_ae)
                    except Exception as _:
                        print(f"Error plotting {edr_file}")
                        continue

    def save_metadata_csv(self, file_path=None):
        """Save the metadata CSV to disk."""
        if file_path is None:
            file_path = os.path.join(self.drive_path, "Figures", "ssusi_metadata.csv")
        if os.path.exists(file_path):
            prev_meta = pd.read_csv(file_path)
            self.meta_data_csv = pd.concat([prev_meta, self.meta_data_csv], ignore_index=True)
            self.meta_data_csv.drop_duplicates(subset=["Filename"], inplace=True)
        self.meta_data_csv.to_csv(file_path, index=False)
