function param = setParams()

%% ALGORITHM PARAMETERS


param.evalMethod='card';                    %'card' require that 2/3 of individals are correctly matched in a group
                                            %'all'  require that 3/3 of individals are correctly matched in a group (more stricter evaluation)

%multi/single frame
param.numFrames=1;                          %number of frames to analyze (>1 imply multiframe analysis)

%parameter for the 2D histogram
param.hist.n_x=20;                          %number of rows for the frustum descriptor
param.hist.n_y=20;                          %number of columns for the frustum descriptor

%displaying options
param.show.weights=0;                       %show the weight used to condense the similarity matrices
param.show.groups=0;                        %show a figure with the current frame, the decetion and the groundtruth
param.show.frustum=1;                       %show the frustum
param.show.results=1;                       %display the precision/recall/F1-score values

%weight calculation parameters
param.weight.mode='MOLP';                   %the multiframe mode is activated only if param.numFrames>1. Set to:
                                            %'MOLP' (MultiObjectiveLinearProgramming)
                                            %'EQUAL' use equal weights for the frames
                                            %'MAXENTROPY' pick the
                                            %combination that maximize the entropy of the weight


%frustum
param.frustum.length=70; %accoutning to interpersonal distance
param.frustum.aperture=160;
param.frustum.samples=2000;

%affinity matrix parameter
param.sigma=0.6;
param.method='JS';

%dataset parameter
param.framesDir='';
end