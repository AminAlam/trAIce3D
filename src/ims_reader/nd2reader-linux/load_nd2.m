function [Images_stack, Image_info] = load_nd2(FileName)

if ~ischar(FileName)
    error('Error. \nFileName must be a char, not a %s.',class(FileName))
end

Image_info = ND2Info(FileName);
Num = Image_info.numImages;
Images_stack = ND2ReadSingle(FileName, 1:Num);

Images_stack = double(Images_stack);
