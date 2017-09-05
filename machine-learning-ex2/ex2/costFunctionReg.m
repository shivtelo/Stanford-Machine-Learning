function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J=0;
tmp=X*theta;
tmp=sigmoid(tmp);
zz=y.*(-log(tmp))+(1-y).*(-log(1-tmp));
reg_t= ( lambda * sum(theta.*theta) )- lambda * theta(1) * theta(1) ;
reg_t = reg_t/2;
J = J + sum(zz);
J = (J+reg_t);
J=J/m;
%fprintf("reg_t%f...\n",  size(reg_t,1));
%fprintf("%f.....\n",  size(reg_t,2));
grad = zeros(size(theta));
tmp3=tmp-y;
tmp4=tmp3;
for i=1:size(X,2)-1
tmp4=[tmp4 tmp3];
end
reg_tv=( lambda * theta )/m;

reg_tv(1)=0;
%fprintf("reg_tv%f...\n",  size(reg_tv,1));
%fprintf("%f.....\n",  size(reg_tv,2));
%fprintf("theta%f...\n",  size(theta,1));
%fprintf("%f.....\n",  size(theta,2));
tmp2 = (tmp4).*X;
tttmp =sum(tmp2)/m;
%fprintf("tttmp%f...\n",  size(tttmp,1));
%fprintf("%f.....\n",  size(tttmp,2));
grad =  (reg_tv')+(sum(tmp2)/m);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
