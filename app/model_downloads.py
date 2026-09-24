"""Pinned model assets; resumable downloads never replace a file before verification."""
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import requests

from model_registry import available_models, is_cuda


class ModelDownloads:
    MODEL_PACKAGES = ('bf16', 'cuda-bf16', 'cuda-fp8', 'analysis', 'tokens')
    ASSET_KEYS = ('repo', 'revision', 'file', 'path', 'size', 'sha256')

    def __init__(self, root):
        self.root = Path(root)
        self.assets, self.lora_catalog = self.load_assets()
        self.lora_ids = tuple(item['id'] for item in self.lora_catalog)
        self.lock = threading.RLock()
        self.state = {'status': 'idle', 'message': '', 'bytes_per_second': 0}
        self.jobs = {}
        self.busy_paths = set()
        self.link_shared()

    def load_assets(self):
        assets = json.loads((self.root / 'config/model-assets.json').read_text())
        catalog_path = self.root / 'config/lora-assets.json'
        catalog = json.loads(catalog_path.read_text()) if catalog_path.is_file() else []
        for item in catalog:
            asset = {key: item[key] for key in self.ASSET_KEYS}
            asset['package'] = item['id']
            assets.append(asset)
        return assets, catalog

    @property
    def PACKAGES(self):
        return self.MODEL_PACKAGES + self.lora_ids

    @property
    def TARGETS(self):
        return self.PACKAGES + ('all', 'missing', 'loras')

    def link_shared(self):
        for name in ('qwen.tiktoken', 'vae.safetensors'):
            source = self.root / 'model/8bit' / name
            target = self.root / 'model/bf16' / name
            if source.is_file() and not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    target.unlink()
                try:
                    target.symlink_to('../8bit/' + name)
                except OSError:
                    if os.name != 'nt':
                        raise
                    # NTFS hard links need no Developer Mode/admin permission.
                    # Both paths live on the same volume in the model folder.
                    os.link(source, target)

    def present(self, asset):
        path = self.root / asset['path']
        return path.is_file() and path.stat().st_size == asset['size']

    def offered_packages(self):
        """Generation packages this machine lists, plus cover-analysis extras."""
        names = [n for n in available_models() if n in self.MODEL_PACKAGES]
        # hardware.models hides CUDA on Linux boxes without NVIDIA; skip those weights too.
        if sys.platform not in ('darwin', 'win32') and not shutil.which('nvidia-smi'):
            names = [n for n in names if not is_cuda(n)]
        return tuple(names) + ('analysis', 'tokens')

    def selected_assets(self, model):
        if model not in self.TARGETS:
            raise ValueError('Unknown model')
        if model in self.lora_ids:
            return [a for a in self.assets if a.get('package') == model]
        if model == 'loras':
            seen, assets = set(), []
            for pkg in self.lora_ids:
                for asset in self.selected_assets(pkg):
                    if asset['path'] not in seen:
                        seen.add(asset['path'])
                        assets.append(asset)
            return assets
        if model == 'missing':
            seen, assets = set(), []
            for pkg in self.offered_packages():
                for asset in self.selected_assets(pkg):
                    if asset['path'] not in seen:
                        seen.add(asset['path'])
                        assets.append(asset)
            return assets
        shared = {'model/8bit/vae.safetensors', 'model/8bit/qwen.tiktoken'}
        return [a for a in self.assets if not a['path'].startswith('loras/') and (
                model == 'all' or
                (model in ('cuda-bf16', 'cuda-fp8') and a['path'].startswith('model/cuda/')) or
                (model == 'analysis' and not a['path'].startswith(('model/', 'tokens/'))) or
                (model == 'tokens' and a['path'].startswith('tokens/')) or
                (model == 'bf16' and
                 (a['path'].startswith('model/bf16/') or a['path'] in shared)))]

    def package_ready(self, model):
        selected = self.selected_assets(model)
        return bool(selected) and all(self.present(a) for a in selected)

    def snapshot(self):
        with self.lock:
            missing = [a for a in self.assets if not self.present(a)]
            packages = {}
            for model in self.PACKAGES + ('missing', 'loras'):
                selected = self.selected_assets(model)
                absent = [a for a in selected if not self.present(a)]
                job = self.jobs.get(model) or {}
                packages[model] = {'ready': bool(selected) and not absent, 'bytes': sum(a['size'] for a in selected),
                                    'missing_bytes': sum(a['size'] for a in absent),
                                    'status': job.get('status') or '', 'file_bytes': job.get('file_bytes') or 0,
                                    'file_total': job.get('file_total') or 0}
            lora_packages = []
            for item in self.lora_catalog:
                pkg = packages[item['id']]
                lora_packages.append({
                    'id': item['id'], 'name': item['name'], 'description': item['description'],
                    'family': item.get('family') or 'Community', 'notes': item.get('notes') or '',
                    'prompt': item.get('prompt') or '', 'settings': item.get('settings') or {},
                    'trigger': item.get('trigger') or '', 'steps': item.get('steps'),
                    'repo': item.get('repo') or '',
                    # 'style' adapters change the writing (and are picked as the Style LoRA); 'sound' ones only the decoder.
                    'slot': 'sound' if item.get('slot') == 'sound' else 'style',
                    'ready': pkg['ready'], 'bytes': pkg['bytes'], 'missing_bytes': pkg['missing_bytes'],
                })
            return {**self.view(), 'packages': packages, 'lora_packages': lora_packages, 'missing': [a['path'] for a in missing],
                    'missing_bytes': sum(a['size'] for a in missing),
                    'total_bytes': sum(a['size'] for a in self.assets)}

    def view(self):
        """Combined transfer state: one job as itself, several as a summed bar."""
        downloading = [job for job in self.jobs.values() if job.get('status') == 'downloading']
        if len(downloading) == 1:
            return dict(downloading[0])
        if downloading:
            files = [job.get('file') for job in downloading if job.get('file')]
            return {
                'status': 'downloading', 'model': 'multiple',
                'message': str(len(downloading)) + ' packages',
                'file': files[0] if len(files) == 1 else '',
                'file_bytes': sum(job.get('file_bytes') or 0 for job in downloading),
                'file_total': sum(job.get('file_total') or 0 for job in downloading),
                'bytes_per_second': sum(job.get('bytes_per_second') or 0 for job in downloading),
                'file_index': 1, 'file_count': len(downloading), 'transfer_mode': 'sequential',
                'connections': len(downloading),
            }
        failed = [job for job in self.jobs.values() if job.get('status') == 'failed']
        if failed:
            return dict(failed[-1])
        if self.jobs and all(job.get('status') == 'complete' for job in self.jobs.values()):
            if len(self.jobs) == 1:
                return dict(next(iter(self.jobs.values())))
            return {'status': 'complete', 'message': 'Selected components are installed.', 'bytes_per_second': 0}
        return dict(self.state)

    def update(self, *args, **values):
        """``update(**fields)`` patches idle test state; ``update(package, **fields)`` patches one job.

        ``model=`` in kwargs is the package id, matching ``downloads.update(status='downloading', model='bf16')``.
        """
        model = args[0] if args else values.pop('model', None)
        if len(args) > 1:
            raise TypeError('update expected at most one positional package id')
        with self.lock:
            if model is not None:
                job = self.jobs.setdefault(model, {'model': model})
                job.update(values)
                job['model'] = model
            else:
                self.state.update(values)

    def claim(self, assets):
        claimed = []
        with self.lock:
            for asset in assets:
                if asset['path'] in self.busy_paths:
                    continue
                self.busy_paths.add(asset['path'])
                claimed.append(asset)
        return claimed

    def release(self, path):
        with self.lock:
            self.busy_paths.discard(path)

    def start(self, model='bf16'):
        self.selected_assets(model)
        if model == 'loras':
            with self.lock:
                for pkg in self.lora_ids:
                    if self.package_ready(pkg) or self.jobs.get(pkg, {}).get('status') == 'downloading':
                        continue
                    self.jobs[pkg] = {'status': 'downloading', 'model': pkg, 'message': 'Preparing downloads…',
                                      'bytes_per_second': 0, 'file_bytes': 0, 'file_total': 0}
                    threading.Thread(target=self.run, args=(pkg,), daemon=True).start()
                return self.snapshot()
        with self.lock:
            if self.jobs.get(model, {}).get('status') == 'downloading':
                return self.snapshot()
            self.jobs[model] = {'status': 'downloading', 'model': model, 'message': 'Preparing downloads…',
                                'bytes_per_second': 0, 'file_bytes': 0, 'file_total': 0}
            threading.Thread(target=self.run, args=(model,), daemon=True).start()
            return self.snapshot()

    def run(self, model='all'):
        claimed = []
        try:
            self.update(model, status='downloading', message='Preparing downloads…', bytes_per_second=0, file_bytes=0, file_total=0)
            while True:
                remaining = [a for a in self.selected_assets(model) if not self.present(a)]
                if not remaining:
                    break
                batch = self.claim(remaining)
                if not batch:
                    time.sleep(0.4)
                    continue
                claimed.extend(batch)
                needed = 0
                for a in batch:
                    part = (self.root / a['path']).with_suffix('.part')
                    partial = part.stat().st_size if part.exists() else 0
                    needed += a['size'] - min(partial, a['size'])
                if shutil.disk_usage(self.root).free < needed + 1024**3:
                    raise RuntimeError(f'Not enough free disk space. Free at least {needed / 1e9 + 1:.1f} GB and retry.')
                for index, asset in enumerate(batch, 1):
                    self.update(model, file=asset['path'], file_index=index, file_count=len(batch))
                    try:
                        self.download(asset, model)
                    finally:
                        self.release(asset['path'])
                        claimed = [a for a in claimed if a['path'] != asset['path']]
                    self.link_shared()
                waiting = [a for a in remaining if a['path'] not in {b['path'] for b in batch} and not self.present(a)]
                if not waiting:
                    break
            self.link_shared()
            self.update(model, status='complete', message='Selected components are installed.', bytes_per_second=0)
        except Exception as exc:
            for asset in claimed:
                self.release(asset['path'])
            self.update(model, status='failed', message=str(exc), bytes_per_second=0)

    def download(self, asset, model=None):
        target = self.root / asset['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix('.part')
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > asset['size']:
            partial.unlink()
            offset = 0
        self.update(model, message='Downloading ' + asset['path'], file=asset['path'], file_bytes=offset, file_total=asset['size'])
        if offset < asset['size']:
            url = f"https://huggingface.co/{asset['repo']}/resolve/{asset['revision']}/{asset['file']}"
            self.sequential_download(url, partial, offset, asset['size'], model)
        self.verify(asset, partial, target, model)

    def sequential_download(self, url, partial, offset, total, model=None):
        self.update(model, transfer_mode='sequential', connections=1)
        with requests.get(url, headers={'Range': f'bytes={offset}-'} if offset else {},
                          stream=True, timeout=(20, 60)) as response:
            response.raise_for_status()
            if response.status_code == 206:
                expected = f'bytes {offset}-'
                if not response.headers.get('Content-Range', '').startswith(expected):
                    raise RuntimeError('Unexpected download range. Retry the download.')
            elif response.status_code == 200:
                offset = 0  # Server ignored Range; safely restart the partial file.
            else:
                raise RuntimeError(f'Unexpected download response: {response.status_code}')
            initial = offset
            started = time.monotonic()
            with partial.open('ab' if offset else 'wb') as output:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    if offset + len(chunk) > total:
                        raise RuntimeError('Download exceeded its expected size.')
                    output.write(chunk)
                    offset += len(chunk)
                    self.update(model, file_bytes=offset, bytes_per_second=(offset - initial) / max(.01, time.monotonic() - started))

    def verify(self, asset, partial, target, model=None):
        self.update(model, message='Verifying ' + asset['path'], bytes_per_second=0)
        digest = hashlib.sha256()
        with partial.open('rb') as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        if partial.stat().st_size != asset['size'] or digest.hexdigest() != asset['sha256']:
            partial.unlink()
            raise RuntimeError('Download verification failed for ' + asset['path'] + '. Retry to download it again.')
        partial.replace(target)
