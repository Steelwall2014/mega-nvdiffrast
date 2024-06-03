from collections import deque
import torch
import tensor_pool as tp
import renderutils as ru

from functools import _CacheInfo, _make_key

PREV = 0
NEXT = 1
KEY = 2
RESULT = 3
class deffered_lru_cache:
    def __init__(self, user_function, maxsize=128):
        self.cache = {}
        self.hits = self.misses = 0
        self.full = False
        self.cache_get = self.cache.get
        self.cache_len = self.cache.__len__
        self.root = []
        self.root[:] = [self.root, self.root, None, None]
        self.user_function = user_function
        self.maxsize = maxsize

    def get(self, *args, **kwargs):
        key = _make_key(args, kwargs, False)
        link = self.cache_get(key)
        if link is not None:
            link_prev, link_next, _key, result = link
            link_prev[NEXT] = link_next
            link_next[PREV] = link_prev
            last = self.root[PREV]
            last[NEXT] = self.root[PREV] = link
            link[PREV] = last
            link[NEXT] = self.root
            self.hits += 1
            return result
        self.misses += 1
        result = self.user_function(*args, **kwargs)
        if key in self.cache:
            pass
        elif self.full:
            """
            由于是"deffered", 所以不会直接删除, 而是需要使用者自行调用remove_from_end
            """
            pass
            # oldroot = self.root
            # oldroot[KEY] = key
            # oldroot[RESULT] = result
            # root = oldroot[NEXT]
            # oldkey = root[KEY]
            # oldresult = root[RESULT]
            # root[KEY] = root[RESULT] = None
            # del self.cache[oldkey]
            # self.cache[key] = oldroot
        else:
            last = self.root[PREV]
            link = [last, self.root, key, result]
            last[NEXT] = self.root[PREV] = self.cache[key] = link
            self.full = (self.cache_len() >= self.maxsize)
        return result

    def cache_info(self):
        return _CacheInfo(self.hits, self.misses, self.maxsize, self.cache_len()) 

    def cache_clear(self):
        self.cache.clear()
        link = self.root[NEXT]
        while link is not self.root:
            next_link = link[NEXT]
            del link[:]
            link = next_link
        self.root[:] = [self.root, self.root, None, None]
        self.hits = self.misses = 0
        self.full = False

    def remove_from_end(self):
        if self.root[PREV] is self.root:
            return None, None
        link = self.root[NEXT]
        link_prev, link_next, key, result = link
        link_next[PREV] = self.root
        self.root[NEXT] = link_next
        del self.cache[key]
        return key, result   

    def __call__(self, *args, **kwargs):
        return self.get(*args, **kwargs)
    
    def items(self):
        it = self.cache.items()
        for k, v in it:
            yield k, v[RESULT]

class CPUOffloadCache:

    def __init__(self, lru_cache_max_size: int, factory_fn) -> None:

        self._async_add_queue = deque()
        self._async_add_handle = None
        self._async_add_input, self._async_add_other = None, None
        self._factory_fn: deffered_lru_cache = deffered_lru_cache(maxsize=lru_cache_max_size, user_function=factory_fn)
        self._offloaded_grads = {}

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self._factory_fn(*args, **kwargs)

    def get_cached_items(self) -> list[tuple[int, torch.Tensor]]:
        return list(self._factory_fn.items())
    
    def get_offloaded_grads(self) -> list[tuple[int, torch.Tensor]]:
        return list(self._offloaded_grads.items())
    
    def cache_clear(self):
        self._factory_fn.cache_clear()

    def offload_clear(self):
        for key, grad in self._offloaded_grads.items():
            if grad is not None:
                tp.release(grad)
        self._offloaded_grads.clear()

    def offload(self):
        pages_cuda_grads: list[tuple[int, tuple]] = []
        maxsize = self._factory_fn.cache_info().maxsize
        while self._factory_fn.cache_info().currsize > maxsize:
            key, cuda_grad = self._factory_fn.remove_from_end()
            if cuda_grad is not None:
                pages_cuda_grads.append((key, cuda_grad))
        if len(pages_cuda_grads) == 0:
            return
        
        with torch.no_grad():

            add_input = []
            add_other = []
            for key, cuda_grad in pages_cuda_grads:
                cpu_grad = tp.to_cpu(cuda_grad)
                if key in self._offloaded_grads:
                    existing_grad = self._offloaded_grads[key]
                    if existing_grad is not None:
                        add_input.append(existing_grad)         # 先前就有grad，那就相加
                        add_other.append(cpu_grad)
                    else:
                        self._offloaded_grads[key] = cpu_grad   # 先前没有grad，现在有grad
            if len(add_input) > 0:
                self._async_add_queue.append((add_input, add_other))

            if self._async_add_handle is None or self._async_add_handle.done():
                if self._async_add_other is not None:
                    for other in self._async_add_other:
                        tp.release(other)
                self._async_add_handle, self._async_add_input, self._async_add_other = None, None, None
                if len(self._async_add_queue) > 0:
                    add_input, add_other = self._async_add_queue.popleft()
                    self._async_add_handle = ru.async_add_(add_input, add_other)
                    self._async_add_input, self._async_add_other = add_input, add_other   # 要保持引用吗？

    def wait_for_offload(self):
        if self._async_add_other is not None:
            for other in self._async_add_other:
                tp.release(other)
        if self._async_add_handle is not None:
            self._async_add_handle.join()
            self._async_add_handle, self._async_add_input, self._async_add_other = None, None, None
        for add_input, add_other in self._async_add_queue:
            for i in range(len(add_input)):
                add_input[i].add_(add_other[i])
                tp.release(add_other[i])