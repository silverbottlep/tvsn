function gen_vis_maps(img_dir)
	addpath('./MatlabEXR')
	%img_dir = '../ObjRenderer/test/';

	listing = dir(img_dir);
	listing = listing(3:end);
	n_models = size(listing,1);

	% compute projection matrix
	fov = 30;
	near = 0.01;
	far = 100;
	e = 1/tan(fov*pi/180/2);
	proj_mat = [e,0,0,0; 
	0,e,0,0;
	0,0,-(far+near)/(far-near),-(2*far*near)/(far-near);
	0,0,-1,0;];

	plot = 0;
	H=256; W=256;

	for i=1:n_models
		for phi = 0:10:20
			path = [fullfile(img_dir,listing(i).name) '/model_views/'];
			out_map_file = [path sprintf('maps_%d.mat',phi)];
			t_start = tic;
			out_map = zeros(64,64,18,18);
			j=1;

			for in_theta=0:20:340
				in_coord_file = [path sprintf('%d_%d_coord.exr',in_theta,phi)];
				in_norm_file = [path sprintf('%d_%d_norm.exr',in_theta,phi)];
				temp = imresize(exrread(in_coord_file),[H,W]); 
				%% change axis, x,y,z -> z,y,x
				in_coord(:,:,1) = temp(:,:,3); 
				in_coord(:,:,2) = temp(:,:,2); 
				in_coord(:,:,3) = temp(:,:,1);
				in_coord = reshape(in_coord,H*W,3);
				temp = imresize(exrread(in_norm_file),[H,W]); 
				in_norm(:,:,1) = temp(:,:,3); 
				in_norm(:,:,2) = temp(:,:,2); 
				in_norm(:,:,3) = temp(:,:,1);
				in_norm = reshape(in_norm,H*W,3);

				homo_coord = zeros(H*W,3+1);
				k=1;

				if plot>0
					in_im = imread([path sprintf('%d_%d.png',in_theta,phi)]);
					figure(1);
					subplot(4,5,k); imagesc(in_im);
				end

				for theta=20:20:340
					% rotate input normal
					homo_coord(:,1:3) = in_norm;
					homo_coord(:,4)=1;
					bases = [0;0;1];
					rot_mat = rotate_matrix(in_theta-90+theta,phi);
					rot_norm = (rot_mat*homo_coord')';
					rot_norm_direct = rot_norm(:,1:3)*bases;
					rot_norm_direct = uint8(rot_norm_direct>0);

					%% camera is 4 unit away from object, 
					%%	meaning push object 4 units away in z direction
					rot_mat = rotate_matrix(in_theta-90+theta,phi);
					rot_mat(3,4) = -4;
					homo_coord(:,1:3) = in_coord;
					homo_coord(:,4)=1;
					pred_coord = (proj_mat*rot_mat*homo_coord')';
					pred_coord = pred_coord./repmat(pred_coord(:,4),1,4);
					%% [-1,1] -> [1,256]
					map = pred_coord(:,1:3)~=0;
					pred_coord = map.*((((pred_coord(:,1:3)+1)/2)*255)+1);
					pred_coord = round(pred_coord);

					%% reshape into 2 dim
					pred_coord = reshape(pred_coord(:,1:3),H,W,3);
					rot_norm_direct = reshape(rot_norm_direct,H,W);

					pred_map = zeros(H,W);
					for h=1:H
						for w=1:W
							x = pred_coord(h,w,1);
							y = pred_coord(h,w,2);
							if rot_norm_direct(h,w) > 0
								if x>0 && y>0 && x<=W && y <=H
									pred_map(H-y,x+1) = 255;
						end
					end
				end
			end

			% SYMMETRY 
			% rotate input normal
			homo_coord(:,1:3) = in_norm;
			homo_coord(:,4)=1;
			homo_coord(:,3) = -homo_coord(:,3);
			rot_mat = rotate_matrix(in_theta-90+theta,phi);
			rot_norm = (rot_mat*homo_coord')';
			rot_norm_direct = rot_norm(:,1:3)*bases;
			rot_norm_direct = uint8(rot_norm_direct>0);


			rot_mat = rotate_matrix(in_theta-90+theta,phi);
			rot_mat(3,4) = -4;
			homo_coord(:,1:3) = in_coord;
			homo_coord(:,4)=1;
			homo_coord(:,3) = -homo_coord(:,3);
			pred_coord = (proj_mat*rot_mat*homo_coord')';
			pred_coord = pred_coord./repmat(pred_coord(:,4),1,4);

			%% [-1,1] -> [1,256]
			map = pred_coord(:,1:3)~=0;
			pred_coord = map.*((((pred_coord(:,1:3)+1)/2)*255)+1);
			pred_coord = round(pred_coord);

			pred_coord = reshape(pred_coord(:,1:3),H,W,3);
			rot_norm_direct = reshape(rot_norm_direct,H,W);

			for h=1:H
					for w=1:W
						x = pred_coord(h,w,1);
						y = pred_coord(h,w,2);
						z = pred_coord(h,w,3);
						if rot_norm_direct(h,w) > 0
							if x>0 && y>0
								if x<=W && y <=H
									pred_map(H-y,x+1,:) = 255;
							end
						end
					end
				end
			end

			pred_map = imresize(pred_map,[64,64]);
			out_map(:,:,j,k) = pred_map;
			k = k+1;

			if plot>0 
						subplot(4,5,k); imagesc(pred_map);
					end
				end
				j = j+1;
				if plot>0 
					pause;
				end
				in_coord = reshape(in_coord,H,W,3);
				in_norm = reshape(in_norm,H,W,3);
			end
			save(out_map_file,'out_map');
			t_end = toc(t_start);
			fprintf('processed model:%d(phi %d), %f seconds\n', i, phi, rem(t_end,60));
		end % for phi=0:10:20
	end % for i=1:n_models
