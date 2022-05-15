talkdown_strings_true = [
    # Condescending
    # dominance = power
    # arousal = agency
    # valence = sentiment
    
    # dominance: think  0.618, means  0.460
    # arousal: means 0.373, think 0.408, 
    "I do not think that word means what you think it means.",
    
    # dominance: idea  0.704, significant  0.786, retarded  0.161
    # arousal: significant 0.649, game 0.788, break 0.265, retarded 0.462
    "my idea is significantly less game-breakingly retarded than yours.",
    
    # dominance: 
    #     speaker: study  0.768, confirm 0.806, science  0.741
    #     addressee: sorry 0.212, conform 0.375, 
    # arousal: study 0.361, confirm 0.439, sorry 0.362, science 0.429, conform 0.380
    "Many studies confirm the above; I'm sorry that science doesn't conform to your worldview.",
    
    # dominance: fifteen  0.443, year  0.415, old  0.315
    # arousal: fifteen 0.290, year 0.284, old 0.279
    "Oh man you really are a fifteen year old aren't you?",
    
    # dominance: kids  0.417, naive  0.375, fuck  0.446
    # arousal: kids 0.500, naive 0.260, fuck 0.930
    "Some of these kids are naive as fuck.",
    
    # dominance: 
    #     speaker: rich  0.905, nice  0.650
    #     addressee: spend  0.429, cost  0.439, make  0.480
    # arousal: rich 0.627, nice 0.442, spend 0.745, cost 0.420, make 0.420
    "I don't know why he would have to be rich to spend 10k on a ring, my current SO agrees that's what a nice ring costs and he makes 120k",
    
    # dominance: 
    #     speaker: credence 0.737, blunt 0.925
    #     addressee: delusion 0.400
    # arousal: important 0.630, credence 0.460, delusion 0.647, blunt 0.680, give 0.346, 
    "Their lives are important enough to me that I will not give credence to their delusions. (Please don't take that as aggression or disrespect, I'm just speaking bluntly)",
    
    # dominance: ask  0.459, tell  0.457,  surprise  0.562, wont  0.291 
    # arousal: job 0.541, ask 0.440, tell 0.350, surprise 0.875, wont 0.343
    "What is a job you ask? I won't tell you, I will let if be a surprise for when you get one. LOL.",
    
    # dominance: hard 0.616, grasp 0.570
    # arousal: hard 0.708, grasp 0.623
    "Is this really so hard to grasp?",
    
    # dominance: deal 0.543, have 0.593
    # arousal: global	0.520, deal 0.599, have 0.389
    "We live in a globalized world, and you have to deal with it.",
    
    # dominance: debate 0.683, intuitive 0.690, understanding 0.689, sorry 0.212, time  0.609, explain 0.736
    # arousal: debate 0.691, intuitive 0.573, understanding 0.304, sorry 0.362, time 0.288, explain 0.517
    "This isn't really up for debate. This is how it works. If you don't have an intuitive understanding of the math of this, I'm sorry I won't be able to spend any more of my time explaining it to you.",

    # no idea what to annotate here
    # arousal: blame 0.640, mechanism 0.583
    "Victim blaming is not about mechanism ",

    # dominance: thing  0.260, male  0.695, female  0.625, oppression  0.615
    # arousal: thing 0.222, male 0.532, female 0.520, oppression 0.673
    "b\/c male oppression is a thing, & female oppression isn't",

    # dominance: world  0.750, begin  0.763, begun  0.404
    # this one is interesting because begin and begun have very different scores
    # arousal: world 0.394, begin 0.569, begun 0.500
    "As the world began last summer for most of you",  
    
    # this one is pretty neutral I think, but positively labeled in dataset
    # dominance: condescending  0.593, talk  0.594, people  0.500
    # arousal: condescending 0.363, talk 0.346, people 0.400
    "i will talk about condescension when people are being condescending.\n\n"
]

talkdown_strings_false = [
    # Not condescending

    # dominance: pollute  0.480, virtue  0.827, signal  0.464, children  0.435, make  0.480, actual  0.716, message  0.415
    # arousal: 
    "it's just that the ideology is polluted with virtue-signalling children who make it very hard to pick out the actual message.",

    # dominance: cop  0.836, deserving  0.858, time  0.609, respect  0.758, random  0.321, civilian  0.598, stop  0.490, chat  0.346
    # arousal: 
    "I'm not anti - cop or anything like that. But that doesn't mean he's deserving of any more of my time, or respect, than any random civilian who wants to stop and have a bit of a chat.",
   
    # dominance: uncomfortable  0.214, people  0.500, exist  0.788, think  0.618, differently  0.506, practically  0.588, offended  0.337, separate  0.349
    # arousal: 
    "he is perpetually uncomfortable that people exist who think differently than him, with different priorities that have nothing to do with his. He's practically offended that people are seperate from himself.",
   
    # dominance: clash  0.453, ideal  0.736, humanity  0.727, guide  0.620
    # arousal: 
    "In the theatre we have a clash of ideals on how humanity should be guided. ",
   
    # dominance: telling  0.527, fucking  0.390, lazy  0.096, fix  0.519
    # arousal: 
    "Just telling people they're fucking lazy is not a fix",
   
    # dominance: motivation  0.802, rage  0.658, stupid  0.200, suddenly  0.456, turn  0.482
    # arousal: 
    "His motivations to hate Spider-Man were purely out of rage, but even then it's stupid how he suddenly turns on Spider-Man. ",
   
    # dominance: acting  0.500, pissingmeoff  0.423, know  0.704, job  0.717, do  0.536, professional  0.943
    # arousal: 
    "How do you do anything without pissing a professional off? By not acting like you know more than they do about their job.",
   
    # dominance: feel  0.596, cool  0.781
    # arousal: 
    "I do feel that people just hate on Trump because it's the \"cool\" thing to do",
   
    # dominance: lady  0.602, believe  0.682
    # arousal: 
    "The ladies in book 2 who believe men don't have in part in their productive cycle", # okay this one is a good example of how this might be a negative example right now but if it's in the context of a conversation it would definitely be condescending 
   
    # dominance: throw  0.623, purport  0.684, condemn  0.625, accept  0.565
    # arousal: 
    "Dee-bo's throwing away of the control just further purports that point. He is good for condemning a boat full of people that are bad, and he is accepting his crimes of being bad.",
   
    # dominance: holy  0.690, fuck  0.446, refuse  0.331
    # arousal: 
    "Holy fuck, I am going to use the hell out of this! I have some friends that I get into door-holding arguments with. As in, I get to the door first and hold it open for them, but they refuse to go through because they - A MAN - should be holding it open for me.",
   
    # dominance: insufferable  0.291, jerks  0.395, lovely  0.741
    # arousal: 
    "after a while they stop being lovely tourists and often become insufferable jerks.",
   
    # dominance: average  0.455, potential  0.840
    # arousal: 
    "This sounds like you're tipping your own fedora. Do you also believe you're smarter than the average bear?Everyone has potential, but there's nothing really there to back it up.", # hmm but this sounds pretty condescending to me
   
    # dominance: want  0.598, thinking  0.595, ahead  0.673
    # arousal: 
    "Look if you just want to excoriate me for thinking the demand for electric cars just isnt here yet, then go ahead.",
   
    # dominance: consensus  0.625, pick  0.525, pass  0.362
    # arousal: 
    "I mean clearly Jordan was not any sort of consensus pick since 2 teams passed over him."
]

