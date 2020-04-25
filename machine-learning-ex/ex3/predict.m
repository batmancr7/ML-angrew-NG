function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);
X = [ones(m, 1) X];


% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
tempa1=zeros(size(Theta1,1),1);
a2=zeros(size(Theta2,1),1);
product=0;
product2=0;
val1=0;
val2=0;
for i=1:m
    for j=1:size(Theta1,1)
        for k=1:size(Theta1,2)
            product=product+Theta1(j,k)*X(i,k);
        end
        tempa1(j)=sigmoid(product);
        product=0;
    end
    a1=[1;tempa1];
    for l=1:size(Theta2,1)
        for m=1:size(Theta2,2)
            product2=product2+Theta2(l,m)*a1(m,1);
        end
        a2(l)=sigmoid(product2);
        product2=0;
    end
    [val1,val2]=max(a2);
    p(i)=val2;
end

% =========================================================================


end
