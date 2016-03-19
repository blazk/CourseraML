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

errors = X*theta - y;              %mx1
J = sum(errors.^2) ./ (2*m) + sum(theta(2:end).^2) .* (lambda/(2*m));

gradreg = theta * lambda;          %2x1
gradreg(1) = 0;                    %do not penalize theta_0
grad = (errors'*X .+ gradreg')/m;  %1x2

% =========================================================================

grad = grad(:);                    %2x1

end
