import os
import os.path as osp
import pandas as pd
import argparse as ap
import subprocess as sp


def parse_args():
    description = 'Helper script for downloading and trimming kinetics videos'
    parser = ap.ArgumentParser(description=description)
    parser.add_argument('inpcsv', type=str, help='csv file of annotation')
    parser.add_argument('outdir', type=str, help='dir to save trimmed videos')
    parser.add_argument('--oridir', type=str, default='temp',
                        help='directory to save original videos')
    parser.add_argument('--save-ori', action='store_true',
                        help='whethere to save the original video')
    parser.add_argument('--trim-fmt', type=str, default='%06d',
                        help='the format for the filename of trimmed videos')
    parser.add_argument('--att-time', type=int, default=5,
                        help='times to attempt to download')
    parser.add_argument('--fail-file', type=str, default='fails.csv',
                        help='files to save the entries failed to download')
    parser.add_argument('--num-jobs', type=int, default=1)
    return parser.parse_args()


class KineticsDownloader:
    baseurl = 'https://www.youtube.com/watch?v='

    def __init__(self, cfg):
        self.anfile = cfg.inpcsv
        self.outdir = cfg.outdir
        self.oridir = cfg.oridir
        self.save_ori = cfg.save_ori
        self.trim_fmt = cfg.trim_fmt
        self.num_jobs = cfg.num_jobs
        self.att_time = cfg.att_time
        self.fail_file = cfg.fail_file
        self.ann = pd.read_csv(self.anfile)
        self.lbl2dir = self.build_vidir()
        self.num_vids = self.ann.shape[0]

    def build_vidir(self):
        """Create a directory for each name in the data"""
        if not osp.exists(self.outdir):
            os.mkdir(self.outdir)
        if not osp.exists(self.oridir):
            os.mkdir(self.oridir)
        lbl2dir = {}
        for lblname in self.ann['label'].unique():
            thisdir = osp.join(self.outdir, lblname)
            if not osp.exists(thisdir):
                os.mkdir(thisdir)
            lbl2dir[lblname] = thisdir
        return lbl2dir

    def get_vidname(self, videoid, stt_time, end_time, label):
        basename = '%s-%s-%s.mp4' % (videoid, self.trim_fmt % stt_time,
                                     self.trim_fmt % end_time)
        dirname = self.label2dir[label]
        videoname = osp.join(self.oridir, basename)
        clipname = osp.join(dirname, basename)
        return videoname, clipname

    def download_video(self, vidname, videoid):
        status = False
        url = self.baseurl + videoid
        cmd = ['youtube-dl', '--quiet', '--no-warnings', '-f', 'mp4',
               '-o', '"%s"'%vidname, '"%s"'%url]
        cmd = ' '.join(cmd)
        attempts = 0
        while True:
            try:
                output = sp.check_output(cmd, shell=True, stderr=sp.STDOUT)
            except sp.CalledProcessError as err:
                attempt += 1
                if attempts == self.att_time:
                    return status, err.output
            else:
                break
        status = osp.exists(vidname)
        return status

    def clip_video(self, vidname, clpname):
        status = False
        cmd = ['ffmpeg', -i, '"%s"'%vidname, '-ss', str(stt_time),
               '-t', str(end_time-stt_time), '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1', '-loglevel', 'panic', '"%s"'%clpname]
        cmd = ' '.join(cmd)
        try:
            output = sp.check_output(cmd, shell=True, stderr=sp.STDOUT)
        except subprocess.CalledProcessError as err:
            return status, err.output
        status = os.path.exists(clpname)
        if not self.save_ori:
            os.remove(vidname)
        return status

    def save_fails(self, fails):
        pass

    def build_dataset(self):
        fails = pd.DataFrame()
        for i, row in self.ann.iterrows():
            print(f'Downloading the {i}/{self.num_videos} video', end=' ')
            videoid, label = row['youtube_id'], row['label']
            stt_time, end_time = row['time_start'], row['time_end']
            vidname, clpname = self.get_vidname(videoid, stt_time, end_time, label)
            print(f'its name is {vidname}')
            dstat = self.download_video(videoid, vidname)
            cstat = self.clip_video(vidname, clpname)
            if dstat and cstat:
                print('Success!')
            else:
                print('Failed')
                fails.append(())
        if len(fails) > 0:
           self.save_fails(fails)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    kdl = KineticsDownloader(args)
    print(kdl.ann)
    print(kdl.num_videos)
    print(kdl.num_feats)
