function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J=0;
tmp=X*theta;
tmp=sigmoid(tmp);
zz=y.*(-log(tmp))+(1-y).*(-log(1-tmp));
J = J + sum(zz);
J = J/m;

grad = zeros(size(theta));

%fprintf("grad old%f...\n",  size(grad,1));
%fprintf("%f.....\n",  size(grad,2));

%fprintf("tmp%f...\n",  size(tmp,1));
%fprintf("%f.....\n",  size(tmp,2));

%fprintf("y%f...\n",  size(y,1));
%fprintf("%f.....\n",  size(y,2));

%fprintf("X%f...\n",  size(X,1));
%fprintf("%f.....\n",  size(X,2));

tmp3=tmp-y;
tmp4=tmp3;
tmp4=[tmp4 tmp3];
tmp4=[tmp4 tmp3];

%fprintf("tmp4%f...\n",  size(tmp4,1));
%fprintf("%f.....\n",  size(tmp4,2));

tmp2 = (tmp4).*X;
%fprintf("tmp2%f...\n",  size(tmp2,1));
%fprintf("%f.....\n",  size(tmp2,2));
%fprintf("%f.....\n",  size(tmp2,3));


grad = sum(tmp2)/m;
%fprintf("gradnew%f...\n",  size(grad,1));
%fprintf("%f.....\n",  size(grad,2));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t.a each parameter in theta
%
% Note: grad should have the same dimensions as theta
%








% =============================================================

end
