"""FID and KID evaluation for scene renderings."""

import os
import shutil
import tempfile
import numpy as np
import torch
from cleanfid import fid
from PIL import Image


def resize_image_to_256x256(image_path, output_path):
    """
    Resize image to 256x256. If smaller, pad with white background.
    If larger, crop from center. Also converts black pixels to white.
    
    Args:
        image_path (str): Path to input image
        output_path (str): Path to save resized image
    """
    img = Image.open(image_path)
    
    # Convert to RGB if not already (handles RGBA, grayscale, etc.)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert black pixels (0,0,0) to white pixels (255,255,255)
    img_array = np.array(img)
    black_mask = (img_array[:, :, 0] == 0) & (img_array[:, :, 1] == 0) & (img_array[:, :, 2] == 0)
    img_array[black_mask] = [255, 255, 255]
    img = Image.fromarray(img_array)
    
    width, height = img.size
    target_size = 256
    if width == target_size and height == target_size:
        # Already correct size
        img.save(output_path)
        return
    
    if width < target_size or height < target_size:
        # Need to pad with white background
        new_img = Image.new('RGB', (target_size, target_size), (255, 255, 255))
        
        # Calculate position to paste the image (center it)
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        new_img.paste(img, (paste_x, paste_y))
        new_img.save(output_path)
    else:
        # Need to crop from center
        left = (width - target_size) // 2
        top = (height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(output_path)





class FIDEvaluator:
    """Evaluator for computing FID and KID scores between real and synthesized scenes."""
    
    def __init__(self, device=None, num_iterations=10, num_workers=0, use_dataparallel=False):
        """Initialize the FID evaluator."""
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.num_iterations = num_iterations
        self.num_workers = num_workers
        self.use_dataparallel = use_dataparallel
    
    def prepare_images(self, path_to_images, temp_dir, num_images=None):
        """Prepare images by resizing and saving to temporary directory."""
        image_files = [
            os.path.join(path_to_images, f)
            for f in os.listdir(path_to_images)
            if f.endswith(".png")
        ]
        
        if num_images:
            np.random.shuffle(image_files)
            image_files = np.random.choice(image_files, num_images)
        
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
            
        for i, img_path in enumerate(image_files):
            output_path = os.path.join(temp_dir, f"{i:05d}.png")
            resize_image_to_256x256(img_path, output_path)
            
        return len(image_files)
    
    def evaluate(self, path_to_real_renderings, path_to_synthesized_renderings, 
                 output_directory=None, temp_dir_base=None, verbose=True):
        """Evaluate FID and KID scores."""
        if verbose:
            print(f"Running FID/KID evaluation on {self.device}")
        
        if temp_dir_base is None:
            temp_dir_base = tempfile.gettempdir()
        
        eval_temp_root = tempfile.mkdtemp(prefix="fid_eval_", dir=temp_dir_base)
        temp_real_dir = os.path.join(eval_temp_root, "real")
        temp_fake_dir = os.path.join(eval_temp_root, "fake")
        
        # Prepare real images once
        if verbose:
            print("Preparing real images...")
        num_real_images = self.prepare_images(path_to_real_renderings, temp_real_dir)

        # Locate TorchScript Inception for clean-fid (offline support).
        # Prefer using a shared path directly on compute nodes; if clean-fid API
        # accepts an explicit inception path we will try to pass it. Otherwise
        # fall back to copying to /tmp as clean-fid expects.
        candidates = [
            "/share/home/202230550120/inception-2015-12-05.pt",
            "/share/home/202230550120/DiffSynth-Studio/评估/inception-2015-12-05.pt",
            "/tmp/inception-2015-12-05.pt",
        ]
        found_inception = None
        for p in candidates:
            if os.path.exists(p):
                found_inception = p
                break

        if found_inception:
            if verbose:
                print(f"Using TorchScript Inception at {found_inception}")
        else:
            if verbose:
                print("Warning: TorchScript Inception not found in candidates; clean-fid may attempt to download and fail in offline environments")

        def _compute_fid_and_kid(real_dir, fake_dir, verbose_local=True):
            """Try calling clean-fid compute_fid/compute_kid while preferring
            passing an explicit inception path. If that fails (unexpected kwarg),
            fall back to copying the file to /tmp and calling without extra kwargs.
            """
            # Attempt to pass the inception path using several plausible kwarg names
            if found_inception:
                param_names = [
                    'inception_path', 'inception', 'inception_pt', 'inception_file',
                    'inception_path_pt', 'torchscript_inception'
                ]
                for name in param_names:
                    try:
                        if verbose_local:
                            print(f"Trying clean-fid with kwarg '{name}'={found_inception}")
                        fid_score = fid.compute_fid(
                            real_dir,
                            fake_dir,
                            device=self.device,
                            num_workers=self.num_workers,
                            use_dataparallel=self.use_dataparallel,
                            verbose=verbose_local,
                            **{name: found_inception},
                        )
                        kid_score = fid.compute_kid(
                            real_dir,
                            fake_dir,
                            device=self.device,
                            num_workers=self.num_workers,
                            use_dataparallel=self.use_dataparallel,
                            verbose=verbose_local,
                            **{name: found_inception},
                        )
                        return fid_score, kid_score
                    except TypeError:
                        # compute_fid didn't accept this kwarg name; try next
                        continue
                    except Exception:
                        # Propagate other errors (e.g., runtime errors from clean-fid)
                        raise

            # Fallback: ensure file is at /tmp/inception-2015-12-05.pt and call normally
            if found_inception and os.path.exists(found_inception):
                target = '/tmp/inception-2015-12-05.pt'
                try:
                    if found_inception != target:
                        shutil.copy2(found_inception, target)
                        if verbose_local:
                            print(f"Copied TorchScript Inception from {found_inception} to {target}")
                    else:
                        if verbose_local:
                            print(f"Found TorchScript Inception at {target}")
                except Exception as e:
                    if verbose_local:
                        print(f"Warning: failed to copy TorchScript Inception to {target}: {e}")

            # Final attempt without extra kwargs
            fid_score = fid.compute_fid(
                real_dir,
                fake_dir,
                device=self.device,
                num_workers=self.num_workers,
                use_dataparallel=self.use_dataparallel,
                verbose=verbose_local,
            )
            kid_score = fid.compute_kid(
                real_dir,
                fake_dir,
                device=self.device,
                num_workers=self.num_workers,
                use_dataparallel=self.use_dataparallel,
                verbose=verbose_local,
            )
            return fid_score, kid_score
        
        fid_scores = []
        kid_scores = []
        
        for iteration in range(self.num_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
            
            # Prepare synthetic images for this iteration
            self.prepare_images(path_to_synthesized_renderings, temp_fake_dir, num_real_images)

            # Compute scores (capture full traceback on error to help offline debugging)
            try:
                fid_score, kid_score = _compute_fid_and_kid(temp_real_dir, temp_fake_dir, verbose_local=verbose)
            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                # write detailed error to output_directory if provided
                try:
                    if output_directory:
                        os.makedirs(output_directory, exist_ok=True)
                        with open(os.path.join(output_directory, "fid_kid_error.log"), "w", encoding="utf-8") as f:
                            f.write(tb)
                except Exception:
                    pass
                # re-raise to be caught by caller
                raise
            
            fid_scores.append(fid_score)
            kid_scores.append(kid_score)
            
            if verbose:
                print(f"  FID: {fid_score:.4f}, KID: {kid_score:.4f}")
            
            # Clean up fake directory for next iteration
            if os.path.exists(temp_fake_dir):
                shutil.rmtree(temp_fake_dir)
        
        # Calculate statistics
        fid_mean = sum(fid_scores) / len(fid_scores)
        fid_std = np.std(fid_scores)
        kid_mean = sum(kid_scores) / len(kid_scores)
        kid_std = np.std(kid_scores)
        
        if verbose:
            print(f"Final FID Score: {fid_mean:.4f} ± {fid_std:.4f}")
            print(f"Final KID Score: {kid_mean:.4f} ± {kid_std:.4f}")
        
        # Clean up temporary directories
        for temp_dir in [temp_real_dir, temp_fake_dir, eval_temp_root]:
            if os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass
        
        return {
            'fid_mean': fid_mean,
            'fid_std': fid_std,
            'fid_scores': fid_scores,
            'kid_mean': kid_mean,
            'kid_std': kid_std,
            'kid_scores': kid_scores,
            'num_iterations': self.num_iterations
        }


