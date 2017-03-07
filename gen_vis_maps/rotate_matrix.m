function rot_mat = rotate_matrix(theta,phi)
	t = theta*pi/180;
	p = phi*pi/180;
	rot_mat = zeros(4,4);
	rot_x = [1,0,0,0; 
					0,cos(p),-sin(p),0;
					0,sin(p), cos(p),0;
					0,0,0,1;];
	rot_y = [cos(t),0,sin(t),0;
					0,1,0,0;
					-sin(t),0,cos(t),0;
					0,0,0,1;];
	rot_mat = rot_x*rot_y;
end
