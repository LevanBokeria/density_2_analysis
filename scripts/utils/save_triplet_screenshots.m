function save_triplet_screenshots()

%% Description

% Script to generate screenshots of triplet trials 

% - will load a CSV specifying query, ref1 and ref2 px sizes.
% - will generate the 3 exemplars
% - will save them

%% Define global variables
dbstop if error;

% Check you're in the right directory
home = pwd;

[~,name,~] = fileparts(home);

if ~strcmp('density_2_analysis',name)
    error('please change working directory to ./con_learn/unified_space/');
end

addpath(genpath(home));

curr_space = 'base_funnel_space';
debugMode = 0;

% Flags
saveImg      = 1;
saveImgNames = 0;
%% Load the appropriate subject and condition specific files

saveFolder = 'try_1';

% Make save folders
saveLoc = fullfile(home,'docs','triplet_screenshots',saveFolder);

if ~exist(saveLoc)
    mkdir(saveLoc);
    
    mkdir(fullfile(saveLoc,'query_central'));
    mkdir(fullfile(saveLoc,'query_side'));
end

%% Start PTB

% Setup PTB with some default values
PsychDefaultSetup(2);
Screen('Preference', 'SkipSyncTests', 2);

if debugMode
    PsychDebugWindowConfiguration(0,0.5);
end

ptb_window = 10;
windowRect = [0 0 1280 1024];

if numel(Screen('Screens')) > 1
    whichScreen = 1;
else
    whichScreen = max(Screen('Screens'));
end

try
    info = Screen('GetWindowInfo',ptb_window);
    screenOpen = 1;
catch
    screenOpen = 0;
end

% set colors
black = BlackIndex(whichScreen); % pixel value for black
white = WhiteIndex(whichScreen);

% screen size
if screenOpen == 0
%         [ptb_window, windowRect] = PsychImaging('OpenWindow', whichScreen, white, [], 32, 2,...
%             [], [],  kPsychNeed32BPCFloat);
            [ptb_window, windowRect] = PsychImaging('OpenWindow', whichScreen, 0.5);

end

if ~strcmp(curr_space,'handle_circle_space')
    Screen('BlendFunction', ptb_window, 'GL_SRC_ALPHA', 'GL_ONE_MINUS_SRC_ALPHA');
end

hz = Screen('NominalFrameRate',ptb_window);

if ~debugMode
    HideCursor;
end

%--- Get the KbQueue up and running.
KbQueueCreate();
KbQueueStart();

% Get the size of the on screen window
[screenXpixels, screenYpixels] = Screen('WindowSize', whichScreen);

% Get the centre coordinate of the window
[xCenter, yCenter] = RectCenter(windowRect);

% make screen white
Screen(ptb_window,'FillRect',0.5 );
Screen('Flip',ptb_window);

% Show or hide the mouse
ShowCursor('Arrow');

%% Read the excel file
triplets = readtable(fullfile(home,'docs','choosing_triplets_new_range2.xlsx'));

% Do not use if comment says "removed" or "replaced to balance out"

% triplets(strcmp(triplets.Comment,'removed') | ...
%     strcmp(triplets.Comment,'replaced to balance out'),:) = [];

nTriplets = height(triplets);

%% Read the inset picture
[inset, ~, ~] = imread(fullfile(home,'docs',...
    'density_space_1.png'));

inset_idx = Screen('MakeTexture', ptb_window, inset);

%% Run main trials
nOptions = 3;

op_x_loc = [300,700,1100]-40;
op_y_loc = [400,400,400];

for iTriplet = 1:nTriplets
 
    % Write out the text
    Screen('TextSize', ptb_window, 20);
    
    % Draw 1 1 white rectangle
    box_rect = [0 0 1200 450];
    box_rect_centered = CenterRectOnPoint(...
        box_rect,screenXpixels/2,screenYpixels/2+200);
    
    Screen('FillRect', ptb_window, [1 1 1], box_rect_centered);
    
    for iOption = 1:nOptions
    
        % Display the stimuli
        if iOption == 1
            xPos1_val = triplets.ref1(iTriplet);            
        elseif iOption == 2
            xPos1_val = triplets.query(iTriplet);
        elseif iOption == 3
            xPos1_val = triplets.ref2(iTriplet);
        end
        yPos1_val = 100;
                
        switch curr_space

            case 'neck_legs_space'

                [rBody, rNeck, rHead, rLegs, rFeet] ...
                    = draw_neck_legs(SF_all,SF_body,...
                    xPos1_val,yPos1_val,...
                    op_x_loc(iOption),op_y_loc(iOption));
                Screen('DrawTextures', ptb_window, ...
                    [Head_idx, Neck_idx Body_idx Legs_idx Feet_idx], [], ...
                    [rHead; rNeck; rBody; rLegs; rFeet]');
                
            case 'base_funnel_space'
                [object, ~, alpha_head] = imread(...
                    fullfile(home,'docs','object-9_120-levels_1D',['object9F0Level' int2str((xPos1_val - 1)) 'F1Level40.png'])); 
                object(:,:,4) = alpha_head; %#ok<*ASGLU> 
                Object_idx = Screen('MakeTexture', ptb_window, object);
                
                rObject = [0,0,500,500];
                rObject = CenterRectOnPoint(rObject,op_x_loc(iOption),710);
                
                Screen('DrawTexture',ptb_window,[Object_idx], [], [rObject]');
                
        end % switch

        % Add the text
        DrawFormattedText(ptb_window, int2str(xPos1_val), op_x_loc(iOption),...
            op_y_loc(iOption)-420, [1 1 1]);
        DrawFormattedText(ptb_window, ['sim ' triplets.simulationName{iTriplet}], ...
            50,50, [1 1 1]);        
%         DrawFormattedText(ptb_window, ['alpha = ' int2str(triplets.alpha(iTriplet))], ...
%             50,100, [1 1 1]);    
        
        fileName = ['triplet_' int2str(iTriplet) ...
            '_template_' int2str(triplets.dist_query_ref1(iTriplet)) '_' ...
            int2str(triplets.dist_query_ref2(iTriplet)) '_' ...
            int2str(triplets.dist_ref1_ref2(iTriplet)) ...
            '_query_' int2str(triplets.query(iTriplet))...
            '_ref1_' int2str(triplets.ref1(iTriplet))...
            '_ref2_' int2str(triplets.ref2(iTriplet))...
            '.png'];
        
        DrawFormattedText(ptb_window, fileName, ...
            70,470, [1 1 1]);            
        
        
    end
    
    %% Add an inset of the density image
    inset_width = 600;
%     inset_height = inset_width*3352/3944;
    inset_height = inset_width*747/1042;
    
    inset_x_center = screenXpixels/2+50;
    inset_y_center = 220;
    
    rinset = CenterRectOnPoint([0 0 inset_width inset_height], ...
        inset_x_center, inset_y_center);
    
    Screen('DrawTextures', ptb_window, ...
        [inset_idx], [], ...
        [rinset]');
    
    % Draw the query and refs on the density map
    inset_scale = 130;
    
    inset_left_edge_to_30 = 78; % How many pixels from the left corner to the 30th x tick?
    inset_right_edge_to_120 = 35;
    inset_30_to_120_width = inset_width - inset_left_edge_to_30 - inset_right_edge_to_120;
    inset_unit_pixel_value = inset_30_to_120_width/90;
    
    % Query
    inset_query_x = inset_x_center - inset_width/2 + ...
        inset_left_edge_to_30 + ...
        (triplets.query(iTriplet)-30)*inset_unit_pixel_value;
    inset_query_y = inset_y_center-6;
    
    query_rect = CenterRectOnPoint([0 0 4 inset_height-80],inset_query_x,inset_query_y);
    Screen('FillRect', ptb_window, [1 0 0], query_rect);
    
    % ref1
    inset_ref1_x = inset_x_center - inset_width/2 + ...
        inset_left_edge_to_30 + ...
        (triplets.ref1(iTriplet)-30)*inset_unit_pixel_value;
    
    ref1_rect = CenterRectOnPoint([0 0 4 inset_height-80],inset_ref1_x,inset_query_y);
    Screen('FillRect', ptb_window, [0 0 0], ref1_rect);
    
    % ref2
    inset_ref2_x = inset_x_center - inset_width/2 + ...
        inset_left_edge_to_30 + ...
        (triplets.ref2(iTriplet)-30)*inset_unit_pixel_value;
    
    ref2_rect = CenterRectOnPoint([0 0 4 inset_height-80],inset_ref2_x,inset_query_y);
    Screen('FillRect', ptb_window, [0 0 0], ref2_rect);
    
    pause(0.05);
    
%     Screen('Flip',ptb_window);    
    
    %% Save a screenshot of the trials
        
    % GetImage call. Alter the rect argument to change the location of the screen shot
    rect_to_save = [1,1,screenXpixels-1,screenYpixels-1];
    
    imageArray = Screen('GetImage', ptb_window, rect_to_save, 'backBuffer');
    
    Screen('Flip',ptb_window);
    
    if saveImg
        
        subfolder = triplets.trial_type{iTriplet};
        
        % imwrite is a Matlab function, not a PTB-3 function
        imwrite(imageArray, fullfile(saveLoc,subfolder,...
            fileName));
    end
    
    if saveImgNames
        
        curr_str = ['img/neck_legs_space/' ...
            'dim_' int2str(iDim) ...
            '_stim_' int2str(iTriplet) ...
            '_x_' num2str(round(xPos1_val,2)) '_y_' ...
            num2str(round(yPos1_val,2)) '.png'];
        
        stimPaths{iTriplet,1} = curr_str;
    end
    pause(0.05);
end % iStims
% KbWait 
% sca;

end