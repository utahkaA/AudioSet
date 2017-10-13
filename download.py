#! /usr/bin/env python
#! -*- coding: utf-8 -*-
import sys
import os
import os.path as path
import errno
import glob
import shlex
import subprocess
import multiprocessing
from multiprocessing import Pool
from multiprocessing import Process
from collections import namedtuple

import pandas as pd
import youtube_dl as yt

UIColor = namedtuple("color", "BOLD ERROR SUCCESS WARNING")
color = UIColor('\033[1m{}\033[0m',
                '\033[91m{}\033[0m',
                '\033[92m{}\033[0m',
                '\033[93m{}\033[0m')

class Downloader(object):
    def __init__(self, media, n_cpu):
        self._checkdir(media)
        self.max_proc = round(n_cpu)
        self.not_available_mov = list()
        self.n_downloaded_mov = 0
        self.url = "https://www.youtube.com/watch?v={0}"

        # Load movie list whose record has: YouTube id; start senconds; end seconds; and labels.
        mov_cols = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
        self.mov_list = pd.read_csv("./list/unbalanced_train_segments.csv",
                                    header=None, comment='#',
                                    names=mov_cols,
                                    sep=',', skipinitialspace=True)
        self.n_samples = self.mov_list.shape[0]

        # Load label list which shows the relationship between the 'mid' and class names.
        self.label_table = pd.read_csv("./list/class_labels_indices.csv",
                                       usecols=["mid", "display_name"])

    def _checkdir(self, media):
        # Path to media to store AudioSet.
        if path.exists(media):
            self.media = path.abspath(media)
        else:
            raise FileNotFoundError(errno.ENOENT,
                                    os.strerror(errno.ENOENT),
                                    self.media)

        self.path_to_tmp = "{0}/AudioSet/tmp".format(self.media)
        self.path_to_wav = "{0}/AudioSet/wav".format(self.media)

        if not path.isdir(self.path_to_tmp):
            os.mkdir(self.path_to_tmp)
        else:
            tmp_files = glob.glob("{0}/*".format(self.path_to_tmp))
            self.tmp_files = list(map(path.basename, tmp_files))

        if not path.isdir(self.path_to_wav):
            os.mkdir(self.path_to_wav)
        else:
            wav_files = glob.glob("{0}/*".format(self.path_to_wav))
            self.wav_files = list(map(path.basename, wav_files))

    def _split(self):
        max_proc = self.max_proc
        n_samples = self.n_samples

        element_size = n_samples // max_proc
        remainder_size = n_samples % max_proc
        starts = [i * element_size for i in range(max_proc)]
        ends = [start + element_size for start in starts]
        if not remainder_size == 0:
            ends[-1] += remainder_size
        split_range = [(start, end) for start, end in zip(starts, ends)]

        mov_lists = list()
        for r in split_range:
            splited_df = self.mov_list.iloc[r[0]:r[1], :]
            splited_df = splited_df.reset_index(drop=True)
            mov_lists.append(splited_df)
        return mov_lists

    def start(self):
        mov_lists = self._split()

        jobs = list()
        for mov_list in mov_lists:
            proc = multiprocessing.Process(target=self._start,
                                           args=(mov_list,))
            jobs.append(proc)
            proc.start()

        for job in jobs:
            job.join()

    def _start(self, mov_list):
        for i, mov_record in mov_list.iterrows():
            self.n_downloaded_mov += 1
            msg = "Progress: {0}/{1}".format(self.n_downloaded_mov,
                                             self.n_samples)
            print(color.BOLD.format(msg))

            ytid = mov_record['YTID']
            start = int(mov_record['start_seconds'])
            end = int(mov_record['end_seconds'])

            target_name = "{0}.wav".format(ytid)
            if target_name in self.wav_files:
                # Remove tmp file
                self._remove_temp(ytid)
                continue
            else:
                ydl_opts = {
                        "format": "bestaudio/best",
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "wav"
                        }],
                        "outtmpl": "{0}/%(id)s.%(ext)s".format(self.path_to_tmp)
                }
                try:
                    # Download a target Youtube movie.
                    with yt.YoutubeDL(ydl_opts) as ydl:
                        ydl.download([self.url.format(ytid)])
                    # Make all wav files same format
                    self._sox(ytid)

                    self._trim(ytid, start, end)

                except yt.utils.DownloadError:
                    self.not_available_mov.append(ytid)

                # Remove tmp file
                self._remove_temp(ytid)

    def _sox(self, ytid):
        cmd = "sox {0}/{2}.wav {1}/{2}.wav channels 1 rate 16k"
        cmd = cmd.format(self.path_to_tmp,
                         self.path_to_wav,
                         ytid)
        subprocess.call(cmd, shell=True)

    def _trim(self, ytid, start, end):
        duration = int(end) - int(start)
        cmd = "sox {0}/{1}.wav {0}/{1}_trim.wav trim {2} {3}; " \
              "rm {0}/{1}.wav; " \
              "mv {0}/{1}_trim.wav {0}/{1}.wav"
        cmd = cmd.format(self.path_to_wav,
                         ytid,
                         start,
                         duration)
        subprocess.call(cmd, shell=True)

    def _remove_temp(self, ytid):
        tmp = "{0}/{1}.wav".format(self.path_to_tmp, ytid)
        if path.isfile(tmp):
            cmd = "rm {0}".format(tmp)
            cmd = shlex.split(cmd)
            subprocess.call(cmd)

""" TODO
DONE
- ディレクトリがあるかどうか確認
- youtube_dl の outtmpl オプションを調べる
- DL できなかった動画のリストを作成
- SoX でフォーマットを揃える
- SoX でトリミングする
- 並列処理を加える (start)
- tmp を消す処理

CPU
4 Core 8 Thread
"""

if __name__ == "__main__":
    media = sys.argv[1]

    # check number of CPUs.
    n_cpu = multiprocessing.cpu_count()

    downloader = Downloader(media, n_cpu)
    downloader.start()
