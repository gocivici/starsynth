(
(
f = { |msg, time, addr|
    if(msg[0] != '/status.reply') {
        "time: % sender: %\nmessage: %\n".postf(time, addr, msg);
    }
};
thisProcess.addOSCRecvFunc(f);
);

(
OSCdef(\note_C,
    {
    // a simple function that triggers an envelope
		x = {COsc.ar(261.63)!2 * EnvGen.ar(Env.perc,doneAction:2)};
		{FreeVerb.ar(x,0.3,0.7,0.1)}.play
}, '/note_C')
);

(
OSCdef(\note_Db,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(277.18)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_Db')
);

(
OSCdef(\note_D,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(293.66)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_D')
);

(
OSCdef(\note_Eb,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(311.13)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_Eb')
);

(
OSCdef(\note_E,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(329.63)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_E')
);

(
OSCdef(\note_F,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(349.23)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_F')
);

(
OSCdef(\note_Gb,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(369.99)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_Gb')
);

(
OSCdef(\note_G,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(392.00)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_G')
);

(
OSCdef(\note_Ab,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(415.30)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_Ab')
);

(
OSCdef(\note_A,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(440.00)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_A')
);

(
OSCdef(\note_Bb,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(466.16)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_Bb')
);

(
OSCdef(\note_B,
    {
    // a simple function that triggers an envelope
         {SinOsc.ar(493.88)!2 * EnvGen.ar(Env.perc,doneAction:2)}.play;
		//{FreeVerb.ar(x, 0.7, 0.8, 0.25)}.play;
}, '/note_B')
);
)
