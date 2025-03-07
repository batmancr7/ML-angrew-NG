function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
b=size(theta,1);


for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    zerob=zeros(b,1);
    for j=1:b
            for i=1:m  
            zerob(j)=zerob(j)+(X(i,:)*theta-y(i))*X(i,j);
            end
            
    end
    
    for a=1:b
        
        theta(a)= theta(a)-((alpha/m)*zerob(a));
        
    end

      
      
            
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
J_history
end