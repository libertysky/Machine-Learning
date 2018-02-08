function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test = [.01, .03, .1, 1,3,10,30]; 
len = length(test);

obj = ones(1,3);


C=.01; sigma = .01;
for i=1:len
  C= test(i);
  for j = 1:len
      sigma = test(j);
       
       model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
       predictions = svmPredict(model, Xval);
       
       err = mean(double(predictions) ~=yval);
       
       if err < obj(1,1)
         obj(1,1) = err;
         obj(1,2) = C;
         obj(1,3) = sigma;
        end
             
  end
end;

C = obj(1,2);
sigma = obj(1,3);

% =========================================================================

end
