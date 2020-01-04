import torch


class Video:

    def __init__(self, K):
        self.K = K


class Proposal:

    def __init__(self, stt, end):
        self.stt = stt
        self.end = end
        self.dur = end - stt
