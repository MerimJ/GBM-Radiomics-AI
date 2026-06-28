python -c "
import SimpleITK as sitk
img = sitk.ReadImage('data/raw/cfb_gbm/001/t0/1_t0_t1gd.nii.gz')
print('Size:', img.GetSize())
print('Spacing:', img.GetSpacing())
"