function Y = custom_conv2(X, PSF, BC)

p = (size(PSF, 1) - 1) / 2;
X = padarray(X, [p p], 'symmetric');
Y = conv2(X, PSF, 'valid');

end