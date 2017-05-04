The Su dataset can be used for the following tasks:
1) multipitch estimation
2) note tracking
3) streaming

Annotation:
The frame-level and note-level ground truth are compiled from the MIDI files into MIREX format:
	frame-level annotation: in \gt_F0
	note-level annotation: in \gt_Note

MIDI:
By annotating every piece of music part by part, the notes of every source is also stored in the MIDI files, named as:
	<piece name>_<instrument name>.mid
The frame-level and note-level ground truth is compiled from the combined MIDI files from every sources:
	<piece name>_tutti.mid

How to cite:
If this dataset is used in your work, please cite the following paper:
[1] Li Su and Yi-Hsuan Yang, "Escaping from the Abyss of Manual Annotation: New Methodology of Building Polyphonic Datasets for Automatic Music Transcription," in Int. Symp. Computer Music Multidisciplinary Research (CMMR), June 2015.