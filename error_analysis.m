%% read in data
clc;
clear;
close all

err_dir = '/home/xihan/Myworkspace/MATLAB_WS/AR_ultrasound/overlay_error.csv';
err_raw = csvread(err_dir);

mk1_err = removeBadData(err_raw(:,1));
mk2_err = removeBadData(err_raw(:,2));
mk3_err = removeBadData(err_raw(:,3));
mk4_err = removeBadData(err_raw(:,4));
mk5_err = removeBadData(err_raw(:,5));
mk6_err = removeBadData(err_raw(:,6));
mk7_err = removeBadData(err_raw(:,7));
mk8_err = removeBadData(err_raw(:,8));

%% plot data
figure
% marker1 
subplot(2,4,1)
plot(0:size(mk1_err)-1,mk1_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 1')

subplot(2,4,2)
plot(0:size(mk2_err)-1,mk2_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 2')

subplot(2,4,3)
plot(0:size(mk3_err)-1,mk3_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 3')

subplot(2,4,4)
plot(0:size(mk4_err)-1,mk4_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 4')

subplot(2,4,5)
plot(0:size(mk5_err)-1,mk5_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 5')

subplot(2,4,6)
plot(0:size(mk6_err)-1,mk6_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 6')

subplot(2,4,7)
plot(0:size(mk7_err)-1,mk7_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 7')

subplot(2,4,8)
plot(0:size(mk8_err)-1,mk8_err,'LineWidth',1)
xlabel('time stamp')
ylabel('pixel-wise error')
title('MARKER 8')
