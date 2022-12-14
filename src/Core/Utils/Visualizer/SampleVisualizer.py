from os.path import join as osPathJoin
from os.path import isfile
from os import listdir
from vedo import Plotter, load

from DeepPhysX.Core.Visualizer.VedoVisualizer import VedoVisualizer


class SampleVisualizer(VedoVisualizer):

    def __init__(self, folder):
        """
        Display all data in a given directory. Use button next and previous to change the displayed object.

        :param str folder: Name of the folder to open
        """
        super(SampleVisualizer, self).__init__()
        # Load samples in the folder
        files = sorted([f for f in listdir(folder) if isfile(osPathJoin(folder, f))
                        and f.endswith('.npz')])
        self.samples = [osPathJoin(folder, f) for f in files]
        self.id_sample = 0
        # Create visualizer
        self.view = Plotter(title='SampleVisualizer', N=1, axes=0, interactive=True, offscreen=False)
        self.view.addButton(fnc=self.showPreviousSample, pos=(0.3, 0.005), states=["previous"])
        self.view.addButton(fnc=self.showNextSample, pos=(0.7, 0.005), states=["next"])
        # Load and show first sample
        self.current_sample = None
        self.loadSample()

    def showPreviousSample(self):
        """
        Called by button click. Select the previous mesh to display.

        :return:
        """
        if self.id_sample > 0:
            self.id_sample -= 1
        else:
            self.id_sample = len(self.samples) - 1
        self.loadSample()

    def showNextSample(self):
        """
        Called by button click. Select the next mesh to display.

        :return:
        """
        if self.id_sample < len(self.samples) - 1:
            self.id_sample += 1
        else:
            self.id_sample = 0
        self.loadSample()

    def loadSample(self):
        """
        Load a sample to vedo plotter

        :return:
        """
        # Clear previous sample in view
        if self.current_sample is not None:
            self.view.clear(self.current_sample)
        # Load next sample from file
        filename = self.samples[self.id_sample]
        view = load(filename)
        self.current_sample = view.actors
        # Show current sample
        self.view.show(self.current_sample)
