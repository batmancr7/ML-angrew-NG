function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
update=0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

   for j=2:length(theta)
       
      update=update+theta(j,1)*theta(j,1);
      
   end
       
       update=(update*lambda)/(2*m);
       

   
for i=1:m
    

   
    J=J+(-y(i,:)*log(sigmoid(X(i,:)*theta)))-((1-y(i,:))*log(1-sigmoid((X(i,:)*theta))));
   

    
end

J=(J/m)+(update);

for j=1:size(grad,1)
for i=1:m
   
    grad(j,1)=grad(j,1)+((sigmoid(X(i,:)*theta)-y(i,:))*X(i,j));  
end
if j==1
    grad(j,1)= grad(j,1)/m+0;
else
    grad(j,1)= (grad(j,1)/(m))+((lambda *theta(j,1))/m);


end
end

% =============================================================

end
