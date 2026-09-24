"""Local workspace names and per-version membership, without moving audio files."""
import json
import secrets

from song_library import write_json


class WorkspaceError(ValueError):
    pass


class Workspaces:
    # The server holds its library lock around reads and mutations.
    def __init__(self, path):
        self.path = path

    def read(self):
        if self.path.exists():
            return json.loads(self.path.read_text())
        return {'workspaces': [{'id': 'default', 'name': 'My Workspace'}], 'memberships': {}}

    def require(self, data, workspace_id):
        workspace = next((w for w in data['workspaces'] if w['id'] == workspace_id), None)
        if workspace is None:
            raise WorkspaceError('Workspace no longer exists. Choose another workspace.')
        return workspace

    def name(self, data, value, exclude=None):
        value = value.strip()
        if not value or len(value) > 80:
            raise WorkspaceError('Enter a workspace name between 1 and 80 characters.')
        if any(w['id'] != exclude and w['name'].casefold() == value.casefold() for w in data['workspaces']):
            raise WorkspaceError('A workspace with that name already exists.')
        return value

    def create(self, name):
        data = self.read()
        workspace = {'id': secrets.token_hex(12), 'name': self.name(data, name)}
        data['workspaces'].append(workspace)
        write_json(self.path, data)
        return workspace

    def rename(self, workspace_id, name):
        data = self.read()
        workspace = self.require(data, workspace_id)
        workspace['name'] = self.name(data, name, workspace_id)
        write_json(self.path, data)
        return workspace

    def delete(self, workspace_id):
        data = self.read()
        self.require(data, workspace_id)
        if workspace_id == 'default':
            raise WorkspaceError('The default workspace cannot be deleted.')
        data['workspaces'] = [w for w in data['workspaces'] if w['id'] != workspace_id]
        # Explicit moves and inherited generation destinations both fall back to default.
        data['memberships'] = {k: ('default' if v == workspace_id else v)
                               for k, v in data['memberships'].items()}
        write_json(self.path, data)

    def move(self, workspace_id, keys):
        data = self.read()
        self.require(data, workspace_id)
        data['memberships'].update(dict.fromkeys(keys, workspace_id))
        write_json(self.path, data)

    def forget(self, job, index, entire_job=False):
        if not self.path.exists():
            return
        data = self.read()
        data['memberships'] = {k: v for k, v in data['memberships'].items()
                               if not (k.startswith(job + ':') if entire_job else k == f'{job}:{index}')}
        write_json(self.path, data)
