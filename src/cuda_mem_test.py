import os

#cuda_alloc_conf = os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
# print("PYTORCH_CUDA_ALLOC_CONF:", cuda_alloc_conf)

# # To specifically check if expandable_segments is True
# if cuda_alloc_conf is not None and 'expandable_segments:True' in cuda_alloc_conf:
#     print("Expandable segments is enabled")
# else:
#     print("Expandable segments is not enabled")
    
    
    
import torch

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name())
    
    # Initialize CUDA by creating a small tensor
    dummy = torch.zeros(1).cuda()
    
    # Now check memory stats again
    memory_stats = torch.cuda.memory_stats()
    print("\nMemory allocator settings after CUDA initialization:")
    print("allocated_bytes:", memory_stats.get('allocated_bytes.all.current'))
    print("reserved_bytes:", memory_stats.get('reserved_bytes.all.current'))
    print("active_bytes:", memory_stats.get('active_bytes.all.current'))
    
    # Additional memory info
    print("\nMemory summary:")
    print(torch.cuda.memory_summary())
else:
    print("CUDA is not available on this system")