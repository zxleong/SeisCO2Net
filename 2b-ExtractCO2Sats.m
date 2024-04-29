clear all
dx = 1.5; %m
dy = 1.5; %m
dz = 1.5; %m
X = 0:dx:32*dx; 
Y = 0:dx:48*dy; 
Z = 0:dz:45*dz;  
G = tensorGrid(X,Y,Z);
G = computeGeometry(G);

% CO2 Saturation
for b = 0:39
    path = 'data/MRST_Outputs/v4_states_f%d.mat';
    sload = load(sprintf(path,b)).states{1,1};
    
    for i=1:48
        newS(i,:,:,:) = reshape(sload{i}.s(:,2),G.cartDims);
    end
    sp = 'data/MRST_Outputs_Proc/s%d.mat';
    save(sprintf(sp,b),'newS');
end

% Pressure
for b = 0:39
    path = 'data/MRST_Outputs/v4_states_f%d.mat';
    sload = load(sprintf(path,b)).states{1,1};
    
    for i=1:48
        newP(i,:,:,:) = reshape(sload{i}.pressure,G.cartDims);
    end
    sp = 'data/MRST_Outputs_Proc/s%d_pres.mat';
    save(sprintf(sp,b),'newP');
end