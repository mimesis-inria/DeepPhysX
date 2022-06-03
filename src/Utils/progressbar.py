from vedo import ProgressBar


class Progressbar(ProgressBar):

    def __init__(self, start, stop, step=1, c=None, title=''):

        ProgressBar.__init__(self, start=start, stop=stop, step=step, c=c, title=title)

        self.percent_int = 0

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start) * 100
        dd = self.stop - self.start
        if dd:
            self.percent /= self.stop - self.start
        else:
            self.percent = 0
        self.percent_int = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent_int / 100 * af))
        br_bk = "\x1b[2m" + self.char_back * (af - nh)
        br = "%s%s%s" % (self.char * (nh - 1), self.char_arrow, br_bk)
        self.bar = self.title + self.char0 + br + self.char1
        if self.percent < 100:
            ps = " " + str(self.percent_int) + "%"
        else:
            ps = ""
        self.bar += ps
