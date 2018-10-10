% Created by Mohit Agarwal on 10/10/2018
% Georgia Institute of Technology
% Copyright 2018 Mohit Agarwal. All rights reserved.

% Converts EEG data from set files to .dat/.lab files for processing
%
% Please follow the following steps:
% 1. Open 'EEGLAB' and load all 15 datasets from "predict future consumer choices" paper
% 2. Run the script
%
% Output is in the following format: S_<sub_id>_<trial_id>.(dat/lab)

sub_id=1;
epoch_id = 1;


for sub_id=1:15
    a = extractfield(ALLEEG(sub_id).epoch, 'eventtype');
    size_data = size(ALLEEG(sub_id).data,3);
    for i=1:size_data
        if(iscell(a(1)))
            temp = cell2mat(a{1,i});
        else
            temp = a(i);
        end
        sub_epoch(i,1) = temp(end);
    end
    for epoch_id = 1:size_data %1:size_data
        mydata = ALLEEG(sub_id).data;
        filename = ['S',num2str(sub_id,'%02d'),'_',num2str(epoch_id,'%03d')];
        dlmwrite(['new_data/',filename,'.dat'],mydata(:,:,epoch_id)',',');
        dlmwrite(['new_data/',filename,'.lab'],sub_epoch(epoch_id,1),',');
    end
end