function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;
pcount=0;
ncount=0;
% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
m=size(y,1);
for i =1:m
    
    if  y(i,:) == 1
        pcount =pcount+1;
    else
        ncount=ncount+1;
    end
end


    pm=zeros(pcount,2);
    am=zeros(ncount,2);
    
    
for i =1:m    
   
    if  y(i,:)== 1
        pm(i,:)=X(i,:);
    else
        am(i,:)=X(i,:);
    end
end



pm = pm(all(pm,2),:);
am = am(all(am,2),:);

pmx=pm(:,1);
pmy=pm(:,2);

plot(pmx,pmy,'k+','LineWidth', 2, 'MarkerSize', 7);
plot(am(:,1),am(:,2),'ko', 'MarkerFaceColor', 'y','MarkerSize', 7);


% =========================================================================



hold off;

end
