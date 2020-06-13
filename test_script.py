import pretty_midi as pyd
import music21 as m21
import os 

track_statistics={}

midi = 'nottingham_mini/A and D.mid'
midi_data = pyd.PrettyMIDI(midi)
print(midi_data.instruments[0].notes[:20])
print(midi_data.instruments[1].notes[:10])