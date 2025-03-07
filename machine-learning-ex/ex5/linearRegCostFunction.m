function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



h=X*theta;
temp=h-y;

temp1=temp'*temp;

temp2=[0;theta(2:end,:)];
temp2=temp2'*temp2;

J = temp1/(2*m) +((temp2*lambda)/(2*m));

grad(1)=1/m*(sum(temp.*X(:,1)));


a=X(:,2:end);
b=temp.*a;
f=sum(b)/m;
g=(1/m)*(X(:,2:end)'*(temp));
e=(lambda/m)*theta(2:end);
grad(2:end)=(g+e);




% =========================================================================

grad = grad(:);

end
