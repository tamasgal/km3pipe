#!/usr/bin/env python
"""Convert a showerfit file to HDF5.

Usage:
    i3shower2hdf5.py INFILE [OUTFILE]
    i3shower2hdf5.py -h | --help

Options:
    -h --help     Show this screen.
"""

from math import isnan, fabs    # todo: replace with numpy?

import h5py
import pandas as pd

# order of these imports is crucial!!!
# 'from icecube import icetray' secretly loads stuff behind the scenes, it seems        # noqa
# very un-pythonic, but eh
# see also http://software.icecube.wisc.edu/documentation/projects/dataclasses/faq-common-probs.html#runtimeerror-extension-class-wrapper-for-base-class-i3frameobject-has-not-been-created-yet     # noqa
from icecube import icetray, dataio    # noqa
from icecube.icetray import I3Module, I3Bool, I3Int
from icecube.dataclasses import I3Double
from I3Tray import I3Tray
# CRUCIAL import this after the ones before
from icecube import antares_common, antares_reader    # noqa
from icecube.gulliver import I3LogLikelihoodFitParams    # noqa

__author__ = "Moritz Lotze"
__copyright__ = "Copyright 2017, Moritz Lotze and the KM3NeT collaboration."
__credits__ = ["Thomas Heid"]
__license__ = "BSD-3"
__maintainer__ = "Moritz Lotze"
__email__ = "mlotze@km3net.de"
__status__ = "Development"


class WriteScalars(I3Module):
    """DUmp all Scalars to disk."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddParameter("Filename", "Name the file to write into.", "foo.h5")
        self.AddParameter("Tablename", "Name of the table.", "/data")
        self.AddOutBox("OutBox")

    def Configure(self):
        self.filename = self.GetParameter("Filename")
        self.tablename = self.GetParameter("Tablename")
        self.store = []

    def Physics(self, frame):
        out = {}
        for k, v in frame.items():
            if k in out:
                continue
            if type(v) not in {I3Double, I3Bool, I3Int}:
                continue
            out[k] = v.value
        self.store.append(out)
        self.PushFrame(frame)

    def Finish(self):
        arr = pd.DataFrame(self.store).to_records(index=False)
        with h5py.File(self.filename, 'w') as h5:
            h5.create_dataset(
                self.tablename,
                data=arr,
                compression="gzip",
                compression_opts=5,
                shuffle=True,
                fletcher32=True
            )


class KeepReconstructed(I3Module):
    """Discard all events which don't have the full fit."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddOutBox("OutBox")

    def Configure(self):
        pass

    def Physics(self, frame):
        if frame.Has(
                "best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues"
        ):    # noqa
            self.PushFrame(frame)


class ReadEventMeta(I3Module):
    """Read Metadata (Event ID) for bookkeeping."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddOutBox("OutBox")

    def Configure(self):
        pass

    def Physics(self, frame):
        event_id = frame.Get("I3EventHeader").EventID
        frame.Put("EventID", I3Int(event_id))
        self.PushFrame(frame)


class ReadLLHValues(I3Module):
    """Read the LLH values of a final fit."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddParameter(
            "LLHParamContainer", "Name of LLH value container", ""
        )
        self.AddOutBox("OutBox")

    def Configure(self):
        self.llh_cont = self.GetParameter("LLHParamContainer")

    def Physics(self, frame):
        try:
            llh_params = frame.Get(self.llh_cont)
            for name, param in llh_params.items():
                try:
                    frame.Put(name, I3Double(param))
                except RuntimeError:
                    continue
        except KeyError:
            pass
        self.PushFrame(frame)


class ReadRecoParticle(I3Module):
    """Read the position, energy, ... of a fit."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddParameter("ParticleName", "Name of the reco particle", "")
        self.AddOutBox("OutBox")

    def Configure(self):
        self.particlename = self.GetParameter("ParticleName")

    def Physics(self, frame):
        try:
            particle = frame.Get(self.particlename)
        except Exception:
            self.PushFrame(frame)
            return
        particle_map = self._read_particle(particle)
        for key, val in particle_map.items():
            name = self.particlename + '_' + key
            frame.Put(name, I3Double(val))
        self.PushFrame(frame)

    def _read_particle(self, particle):
        out = {}
        out["pos_x"] = particle.GetX()
        out["pos_y"] = particle.GetY()
        out["pos_z"] = particle.GetZ()
        out["time"] = particle.GetTime()
        out["theta"] = particle.GetDir().CalcTheta()
        out["phi"] = particle.GetDir().CalcPhi()
        out["energy"] = particle.GetEnergy()
        return out


class Readrlogl(I3Module):
    """Read the Logl of the final fits."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddOutBox("OutBox")

    def Configure(self):
        self.fit_params = [
            "best_FirstDusjOrcaVertexFit_FitParams",
            "best_SecondDusjOrcaVertexFit_FitParams",
            "best_DusjOrcaUsingProbabilitiesFinalFit_FitParams",
        ]

    def Physics(self, frame):
        for fit_param in self.fit_params:
            try:
                rlogl = self.Readrlogl(frame, fit_param)
            except Exception:
                continue
            if isnan(rlogl):
                rlogl = -0.1
            frame.Put(fit_param + "__rlogl", I3Double(rlogl))
        self.PushFrame(frame)

    def Readrlogl(self, frame, FitParameters):
        FitParameters = frame.Get(FitParameters)
        rlogl = FitParameters.rlogl
        return float(rlogl)


class Compare(I3Module):
    """Compare prefit + final fit."""

    def __init__(self, context):
        I3Module.__init__(self, context)
        self.AddParameter("particle_1", "Name of first particle", "")
        self.AddParameter("particle_2", "Name of second particle", "")
        self.AddOutBox("OutBox2")

    def Configure(self):
        self.InputParticleName = [0, 0]
        self.InputParticleName[0] = self.GetParameter("particle_1")
        self.InputParticleName[1] = self.GetParameter("particle_2")

    def Physics(self, frame):
        self.OK = True
        try:
            self.GetParticles(frame)
        except KeyError:
            pass
        self.Check(frame)

    def Check(self, frame):
        for Particle in self.InputParticles:
            self.CheckParticle(Particle, frame)

    def CheckParticle(self, Particle, frame):
        if isnan(Particle.GetPos().X) or isnan(Particle.GetTime()):
            self.OK = False

    def GetParticles(self, frame):
        self.InputParticles = []
        for InputParticleName in self.InputParticleName:
            self.InputParticles.append(frame.Get(InputParticleName))


class Distance(Compare):
    """Spatial Distance between 2 vertices (prefit + final)."""

    def __init__(self, context):
        Compare.__init__(self, context)
        self.AddOutBox("OutBox")

    def Configure(self):
        Compare.Configure(self)

    def Physics(self, frame):
        try:
            Compare.Physics(self, frame)
            if self.OK:
                self.get_vertex()
                self.get_distance()
                frame.Put("Distance", I3Double(self.Distance))
        except IndexError:
            pass
        self.PushFrame(frame)

    def get_vertex(self):
        self.Vertex = []

        for Particle in self.InputParticles:
            pos = Particle.GetPos()
            self.Vertex.append(pos)

    def get_distance(self):
        self.Distance = self.Vertex[0].CalcDistance(self.Vertex[1])


class TimeDistance(Compare):
    """Time Distance between 2 vertices (prefit + final)."""

    def __init__(self, context):
        Compare.__init__(self, context)
        self.AddOutBox("OutBox")

    def Configure(self):
        Compare.Configure(self)

    def Physics(self, frame):
        try:
            Compare.Physics(self, frame)
            if self.OK:
                self.GetTime()
                self.GetTimeDifference()
                frame.Put("TimeDistance", I3Double(self.TimeDifference))
        except IndexError:
            pass
        self.PushFrame(frame)

    def GetTime(self):
        self.Time = []
        for Particle in self.InputParticles:
            self.Time.append(I3Double(Particle.GetTime()).value)

    def GetTimeDifference(self):
        self.TimeDifference = I3Double(
            fabs(float(self.Time[0]) - float(self.Time[1]))
        )


def i3extract(infile, outfile=None):
    """Main event loop"""
    if outfile is None:
        outfile = infile + '.h5'
    tray = I3Tray()
    tray.AddModule('I3Reader', 'i3_reader', filename=infile)
    # tray.AddModule(KeepReconstructed, "event_selector")
    tray.AddModule(ReadEventMeta, 'read_meta')    # grab the event ID
    tray.AddModule(
        Distance,
        "compare_space",
        particle_1="best_FirstDusjOrcaVertexFit_FitResult",
        particle_2="best_SecondDusjOrcaVertexFit_FitResult"
    )
    tray.AddModule(
        TimeDistance,
        "compare_time",
        particle_1="best_FirstDusjOrcaVertexFit_FitResult",
        particle_2="best_SecondDusjOrcaVertexFit_FitResult",
    )
    tray.AddModule(
        ReadRecoParticle,
        'read_particle_first',
        ParticleName='best_FirstDusjOrcaVertexFit_FitResult'
    )
    tray.AddModule(
        ReadRecoParticle,
        'read_particle_second',
        ParticleName='best_SecondDusjOrcaVertexFit_FitResult'
    )
    tray.AddModule(
        ReadRecoParticle,
        'read_particle_proba',
        ParticleName='best_DusjOrcaUsingProbabilitiesFinalFit_FitResult'
    )    # noqa
    tray.AddModule(
        ReadLLHValues,
        'read_llh_chere',
        LLHParamContainer=
        'best_DusjOrcaUsingProbabilitiesFinalFit_FitResult_FinalLLHValues'
    )    # noqa
    tray.AddModule(WriteScalars, 'write_scalars', filename=outfile)
    tray.AddModule('TrashCan', 'dustbin')
    tray.Execute()
    tray.Finish()


def main():
    """Entry point when running as script from commandline."""
    from docopt import docopt
    args = docopt(__doc__)
    infile = args['INFILE']
    outfile = args['OUTFILE']
    i3extract(infile, outfile)


if __name__ == '__main__':
    main()
