function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
update3=0;
update1=0;
update2=0;
update=0;
y_Vec = (1:num_labels)==y;
X = [ones(m,1), X];  % Adding 1 as first column in X  
a1 = X; % 5000 x 401
z2 = a1 * Theta1';  % m x hidden_layer_size == 5000 x 25
a2 = sigmoid(z2); % m x hidden_layer_size == 5000 x 25
a2 = [ones(size(a2,1),1), a2]; % Adding 1 as first column in z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26
z3 = a2 * Theta2';  % m x num_labels == 5000 x 10
a3 = sigmoid(z3); %

for i=1:m
    for k=1:num_labels
        update3 =update3 +(-(y_Vec(i,k))*log(a3(i,k))-((1-y_Vec(i,k))*log(1-a3(i,k))));
    end

end

for j=1:hidden_layer_size
    for k=2:input_layer_size+1
        update2=update2+Theta1(j,k)*Theta1(j,k);
    end
end

for j=1:num_labels
    for k=2:hidden_layer_size+1
        update1=update1+Theta2(j,k)*Theta2(j,k);
    end
end

update=(lambda/(2*m))*(update2+update1);

J=(update3/m)+update;


Y_Vec = (1:num_labels)==y;  % Adding 1 as first column in X  
A1 = X; % 5000 x 401
Z2 = A1 * Theta1';  % m x hidden_layer_size == 5000 x 25
A2 = sigmoid(Z2); % m x hidden_layer_size == 5000 x 25
A2 = [ones(size(A2,1),1), A2]; % Adding 1 as first column in z = (Adding bias unit) % m x (hidden_layer_size + 1) == 5000 x 26
Z3 = A2 * Theta2';  % m x num_labels == 5000 x 10
A3 = sigmoid(Z3); %

DELTA3 = A3-Y_Vec;
temp=[ones(size(Z2,1),1) sigmoidGradient(Z2)];
DELTA2 =(DELTA3*Theta2).*(temp);
DELTA2=DELTA2(:,2:end);

Theta2_grad=1/m*(DELTA3'*A2);
Theta1_grad=1/m*(DELTA2'*A1);


  %Calculating gradients for the regularization
  Theta1_grad_reg_term = (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)]; % 25 x 401
  Theta2_grad_reg_term = (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)]; % 10 x 26
  
  %Adding regularization term to earlier calculated Theta_grad
  Theta1_grad = Theta1_grad + Theta1_grad_reg_term;
  Theta2_grad = Theta2_grad + Theta2_grad_reg_term;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
