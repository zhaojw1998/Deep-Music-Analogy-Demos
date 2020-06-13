from dataloader import MIDI_Loader, MIDI_Render
import numpy as np
import pickle

loader = MIDI_Loader(datasetName = 'Nottingham', minStep = 0.125)
directory = 'nottingham_mini/'
midi_files = loader.load(directory)
midi_files = loader.getNoteSeq()
midi_files = loader.getChordSeq(recogLevel = "Mm")
#midi_files = loader.dataAugment()
_ = loader.writeFile(output = "nottingham_mini_txt/")

render = MIDI_Render(datasetName = 'Nottingham', minStep = 0.125)
midi = render.text2midi('nottingham_mini_txt/A and D.txt', "Mm", 'test.mid')