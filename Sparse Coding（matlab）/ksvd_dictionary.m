function [img_Y] = ksvd_dictionary(img,bb,K,overlap,J,sigma,rc_min,max_coeff)
%  ����һ������img���Լ����ֲ���  ��KSVDѧϰ�ֵ䣬�ؽ�img����
[N,M]=size(img);
NN = ceil((N-bb)/overlap) * overlap + bb;
MM = ceil((M-bb)/overlap) * overlap + bb;

DCT=zeros(bb,sqrt(K));
for k=0:1:sqrt(K)-1
    V=cos([0:1:bb-1]'*k*pi/sqrt(K));
    if k>0
        V=V-mean(V);
    end
    DCT(:,k+1)=V/norm(V);
end
DCT=kron(DCT,DCT);
IMin0 = zeros(NN,MM);
IMin0(1:N,1:M)=img;
%Compute mask and extracting its patches  ������ߣ���ȡ�䲹��
Mask = double(~(IMin0==0));
blkMask = overlap_im2col(Mask, bb, overlap);
% Extracting the noisy image patches
blkMatrixIm = overlap_im2col(IMin0, bb, overlap);
% Inpainting the Patches (K-SVD)  *********��һ����case2��ͬ��KSVD_Inpainting

[Dict, Coeff]=KSVD_Inpainting(DCT,blkMatrixIm,blkMask,sigma,rc_min,max_coeff,J);  %DCT
% Creating the output image
 img_Y= overlap_col2im(Dict*Coeff, blkMask, bb, overlap, size(IMin0));
end

