%% read in data
clc;
clear;
close all

dir_u = '/home/xihan/Myworkspace/MATLAB_WS/AR_ultrasound/angle_u.csv';
dir_v = '/home/xihan/Myworkspace/MATLAB_WS/AR_ultrasound/angle_v.csv';

angle_u_raw = csvread(dir_u);
angle_v_raw = csvread(dir_v);

angle_u = removeBadData(angle_u_raw);
angle_v = removeBadData(angle_v_raw);

% marker 1-8 vs. u
marker1_u = angle_u(angle_u(:,1)==1,:);
marker2_u = angle_u(angle_u(:,1)==2,:);
marker3_u = angle_u(angle_u(:,1)==3,:);
marker4_u = angle_u(angle_u(:,1)==4,:);
marker5_u = angle_u(angle_u(:,1)==5,:);
marker6_u = angle_u(angle_u(:,1)==6,:);
marker7_u = angle_u(angle_u(:,1)==7,:);
marker8_u = angle_u(angle_u(:,1)==8,:);

% marker 1-8 vs. v
marker1_v = angle_v(angle_v(:,1)==1,:);
marker2_v = angle_v(angle_v(:,1)==2,:);
marker3_v = angle_v(angle_v(:,1)==3,:);
marker4_v = angle_v(angle_v(:,1)==4,:);
marker5_v = angle_v(angle_v(:,1)==5,:);
marker6_v = angle_v(angle_v(:,1)==6,:);
marker7_v = angle_v(angle_v(:,1)==7,:);
marker8_v = angle_v(angle_v(:,1)==8,:);


x_v = marker2_v(:,2);
y_v = marker2_v(:,3);
z_v = marker2_v(:,end);

x_u = marker2_u(:,2);
y_u = marker2_u(:,3);
z_u = marker2_u(:,end);

% surface plotting
% hold on 
% xx_v = linspace(min(x_v), max(x_v), 20);
% yy_v = linspace(min(x_v), max(x_v), 20);
% [X_v,Y_v] = meshgrid(xx_v, yy_v);
% Z_v = griddata(x_v,y_v,z_v,X_v,Y_v);
% surf(X_v, Y_v, Z_v)

%% plot rotX, rotY vs. u, v
% marker2 v
figure
subplot(2,3,1)
plot3(x_v,y_v,z_v,'.r')
grid on
xlabel('rot x [deg]')
ylabel('rot y [deg]')
zlabel('v coordinate')

subplot(2,3,2)
plot(x_v,z_v,'.r')
grid on
xlabel('rot x [deg]')
ylabel('v coordinate')

subplot(2,3,3)
plot(y_v,z_v,'.r')
grid on
xlabel('rot y [deg]')
ylabel('v coordinate')

% marker2 u
subplot(2,3,4)
plot3(x_u,y_u,z_u,'.b')
grid on
xlabel('rot x [deg]')
ylabel('rot y [deg]')
zlabel('u coordinate')

subplot(2,3,5)
plot(x_u,z_u,'.b')
grid on
xlabel('rot x [deg]')
ylabel('u coordinate')

subplot(2,3,6)
plot(y_u,z_u,'.b')
grid on
xlabel('rot y [deg]')
ylabel('u coordinate')

%% rot z 
figure
subplot(2,1,1)
plot(marker2_v(:,4), z_v, '.g')
grid on
xlabel('rot z [deg]')
ylabel('v coordinate')

subplot(2,1,2)
plot(marker2_u(:,4), z_u, '.g')
grid on
xlabel('rot z [deg]')
ylabel('u coordinate')
