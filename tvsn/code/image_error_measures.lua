--------------------------------------------------------------------------------
-- Calcul du SSIM
function SSIM(img1, img2)
  --[[
  %This is an implementation of the algorithm for calculating the
  %Structural SIMilarity (SSIM) index between two images. Please refer
  %to the following paper:
  %
  %Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
  %quality assessment: From error visibility to structural similarity"
  %IEEE Transactios on Image Processing, vol. 13, no. 4, pp.600-612,
  %Apr. 2004.
  %

  %Input : (1) img1: the first image being compared
  %        (2) img2: the second image being compared
  %        (3) K: constants in the SSIM index formula (see the above
  %            reference). defualt value: K = [0.01 0.03]
  %        (4) window: local window for statistics (see the above
  %            reference). default widnow is Gaussian given by
  %            window = fspecial('gaussian', 11, 1.5);
  %        (5) L: dynamic range of the images. default: L = 255
  %
  %Output:     mssim: the mean SSIM index value between 2 images.
  %            If one of the images being compared is regarded as
  %            perfect quality, then mssim can be considered as the
  %            quality measure of the other image.
  %            If img1 = img2, then mssim = 1.]]


   if img1:size(1) > 2 then
    img1 = image.rgb2y(img1)
    img1 = img1[1]
    img2 = image.rgb2y(img2)
    img2 = img2[1]
   end



   -- place images between 0 and 255.
   --img1:add(1):div(2):mul(255)
   --img2:add(1):div(2):mul(255)
   img1:mul(255)
   img2:mul(255)

   local K1 = 0.01;
   local K2 = 0.03;
   local L = 255;

   local C1 = (K1*L)^2;
   local C2 = (K2*L)^2;
   local window = image.gaussian(11, 1.5/11,0.0708);

   local window = window:div(torch.sum(window));

   local mu1 = image.convolve(img1, window, 'full')
   local mu2 = image.convolve(img2, window, 'full')

   local mu1_sq = torch.cmul(mu1,mu1);
   local mu2_sq = torch.cmul(mu2,mu2);
   local mu1_mu2 = torch.cmul(mu1,mu2);

   local sigma1_sq = image.convolve(torch.cmul(img1,img1),window,'full')-mu1_sq
   local sigma2_sq = image.convolve(torch.cmul(img2,img2),window,'full')-mu2_sq
   local sigma12 =  image.convolve(torch.cmul(img1,img2),window,'full')-mu1_mu2

   local ssim_map = torch.cdiv( torch.cmul((mu1_mu2*2 + C1),(sigma12*2 + C2)) ,
     torch.cmul((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)));
   local mssim = torch.mean(ssim_map);
   return mssim
end
