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

  a1 = [ones(m, 1) X];
  z2 = a1*Theta1';
  a2 = [ones(m, 1) sigmoid(z2)];
  z3 = a2*Theta2';
  a3 = sigmoid(z3);
  h = a3;
 
 yr = zeros(m,num_labels);
 for i =1:m
  yr(i,y(i))=1;
 end
  
 J= 1/m * sum(sum((-yr.*log(h)-(1-yr).*log(1-h))));

 %reg
 temp1= Theta1(:,2:end);
 temp2 = Theta2(:,2:end);

 tr1 = sum(sum(temp1.^2));
 tr2 = sum(sum(temp2.^2));
 
J+= lambda/(2*m) * (tr1 + tr2) ;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients

grad1 = zeros(size(Theta1));
grad2 = zeros(size(Theta2));

for t=1:m,
  
  hvec = h(t,:)';
  a1vec=a1(t,:)';
  a2vec = a2(t,:)';
  yvec = yr(t,:)';
  
  delta3 = hvec - yvec;
  z2vec = [1;Theta1 * a1vec];
  delta2 = Theta2' * delta3 .* sigmoidGradient(z2vec);
  
  grad1 += delta2(2:end) *a1vec';
  grad2 += delta3 *a2vec';
  
 end
  
Theta1_grad = grad1/m;
Theta2_grad = grad2/m; 

 
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
r1 = [zeros(size(temp1,1),1) temp1];
r2  = [zeros(size(temp2,1),1) temp2];

Theta1_grad += lambda/m * r1;
Theta2_grad += lambda/m * r2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
