import { apiClient } from '@/data/connectors/client';

describe('apiClient.getRequest', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('opts into payload errors and preserves the legacy error response shape', async () => {
    const payload = {
      error: JSON.stringify({
        type: 'ResourcesUnavailableError',
        message: 'No capacity',
      }),
    };
    global.fetch.mockResolvedValue({
      ok: true,
      clone: () => ({ json: async () => payload }),
    });

    const response = await apiClient.getRequest('request/id');

    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining(
        '/api/get?request_id=request%2Fid&return_error_payload=true'
      ),
      expect.any(Object)
    );
    expect(response.ok).toBe(false);
    expect(response.status).toBe(500);
    await expect(response.json()).resolves.toEqual({ detail: payload });
  });
});
