function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
b=size(theta,1);
theta0=theta(1);
theta1=theta(2);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
            update0=0;
            update1=0;
          
            for i=1:m
                
            update0=update0+(((X(i,:)*theta)-y(i))*X(i,1));
            update1=update1+(((X(i,:)*theta)-y(i))*X(i,2));
            end
            
            theta0= theta0-((alpha/m)*update0);
            theta1= theta1-((alpha/m)*update1);
            
            %theta(1)=theta0;
            %theta(2)=theta1;
            
            theta = [theta0;theta1];

            
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
J_history
end
