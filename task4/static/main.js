async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  return res.json();
}

async function uploadFiles(fileInput) {
  const files = fileInput.files;
  if (files.length === 0) return [];
  const form = new FormData();
  for (let f of files) form.append('file', f);
  const resp = await fetch('/upload', { method: 'POST', body: form });
  const j = await resp.json();
  if (j.success) return j.files;
  else { alert('Upload failed'); return []; }
}

function createPlotlyMesh(responseData, containerId) {
    console.log('=== createPlotlyMesh DEBUG ===');
    console.log('Input responseData:', responseData);
    
    // Handle both response formats
    let meshesData;
    
    if (responseData.meshes) {
        // New format: {meshes: [...]}
        meshesData = responseData.meshes;
        console.log('Using meshes array:', meshesData);
    } else if (responseData.mesh) {
        // Old format: {mesh: [...]}  
        meshesData = responseData.mesh;
        console.log('Using mesh array:', meshesData);
    } else if (Array.isArray(responseData)) {
        // Response is the array itself
        meshesData = responseData;
        console.log('Using response as array:', meshesData);
    } else {
        // Unknown format
        console.error('Unknown response format:', responseData);
        document.getElementById(containerId).innerHTML = 
            '<div class="error">Unknown response format for 3D visualization</div>';
        return;
    }
    
    // Error checking
    if (!meshesData || !Array.isArray(meshesData) || meshesData.length === 0) {
        console.error('Invalid meshes data:', meshesData);
        document.getElementById(containerId).innerHTML = 
            '<div class="error">No mesh data received for 3D visualization</div>';
        return;
    }
    
    console.log('Meshes data length:', meshesData.length);
    console.log('First mesh item:', meshesData[0]);
    
    const traces = [];
    const clusterIds = meshesData.map(mesh => mesh.cluster_id);
    const minId = Math.min(...clusterIds);
    const maxId = Math.max(...clusterIds);
    
    console.log('Cluster IDs:', clusterIds);
    console.log('Min ID:', minId, 'Max ID:', maxId);
    
    // Handle case when all clusters have same ID
    const range = maxId - minId;
    const denominator = range > 0 ? range : 1;
    
    meshesData.forEach((mesh, index) => {
        const colorValue = range > 0 ? (mesh.cluster_id - minId) / denominator : 0.5;
        
        console.log(`Mesh ${index}: cluster_id=${mesh.cluster_id}, colorValue=${colorValue}`);
        
        const trace = {
            type: 'mesh3d',
            x: mesh.x,
            y: mesh.y,
            z: mesh.z,
            i: mesh.i,
            j: mesh.j,
            k: mesh.k,
            opacity: 0.7,
            color: colorValue,
            colorscale: 'Viridis',
            flatshading: true,
            name: `Cluster ${mesh.cluster_id}`,
            hoverinfo: 'name',
            colorbar: {
                title: 'Cluster ID',
                tickvals: [0, 0.5, 1],
                ticktext: [minId, Math.round((minId+maxId)/2), maxId]
            }
        };
        
        traces.push(trace);
    });
    
    const layout = {
        title: `3D Segmentation (${meshesData.length} clusters)`,
        scene: {
            camera: {
                eye: { x: 1.5, y: 1.5, z: 1.5 }
            }
        },
        showlegend: true
    };
    
    console.log('Creating Plotly with', traces.length, 'traces');
    Plotly.newPlot(containerId, traces, layout);
    console.log('Plotly visualization completed');
}

// Function to handle 2D data (fallback)
function create2DHeatmap(twoDData, containerId) {
    console.log('Creating 2D heatmap instead of 3D volume');
    
    const trace = {
        type: 'heatmap',
        z: twoDData,
        colorscale: 'Viridis'
    };
    
    const layout = {
        title: '2D Segmentation Heatmap (Only 1 slice available)',
        xaxis: { title: 'X' },
        yaxis: { title: 'Y' }
    };
    
    Plotly.newPlot(containerId, [trace], layout);
}

window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('uploadBtn').onclick = async () => {
    const filesA = document.getElementById('filesA');
    const filesB = document.getElementById('filesB');
    const uploadedA = await uploadFiles(filesA);
    const uploadedB = await uploadFiles(filesB);
    window._uploadedA = uploadedA;
    window._uploadedB = uploadedB;
    alert('Uploaded A:' + uploadedA.length + ' B:' + uploadedB.length);
  };

  document.getElementById('runA').onclick = async () => {
    const files = window._uploadedA;
    if (!files || files.length === 0) { alert('Upload Series A first'); return; }

    const payload = {
      files: files,
      algorithm: document.getElementById('algorithm').value,
      n_clusters: document.getElementById('n_clusters').value,
      eps: document.getElementById('eps').value,
      min_samples: document.getElementById('min_samples').value,
      use_supervoxels: document.getElementById('use_supervoxels').checked,
      supervoxel_count: document.getElementById('supervoxel_count').value,
      spatial_weight: document.getElementById('spatial_weight').value,
      intensity_weight: document.getElementById('intensity_weight').value
    };

    // Clear previous results
    document.getElementById('plotly3d').innerHTML = '<p>Processing Series A...</p>';
    
    const j = await postJSON('/process_series', payload);
    if (j.success) {
      document.getElementById('previewA').innerHTML =
        '<h4>Original (middle slice)</h4><img src="' + j.orig_slice + '" />';
      document.getElementById('previewSeg').innerHTML =
        '<h4>Segmented (middle slice)</h4><img src="' + j.seg_slice + '" />';
      window._seg_path = j.seg_path;

      // Generate 3D preview with error handling - USE MESH ENDPOINT
      try {
        const plotData = await fetch('/generate_3d_mesh', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seg_path: j.seg_path })
        });
        const plotJson = await plotData.json();
        console.log('=== SERIES A 3D RESPONSE ===');
        console.log('3D Mesh Response:', plotJson);
        console.log('Has mesh:', !!plotJson.mesh);
        console.log('Has meshes:', !!plotJson.meshes);
        console.log('Is array:', Array.isArray(plotJson));
        console.log('Response keys:', Object.keys(plotJson));

        if (plotJson.success) {
            console.log('Mesh stats:', plotJson.stats);
            createPlotlyMesh(plotJson, 'plotly3d');  // Pass entire response
        } else {
            document.getElementById('plotly3d').innerHTML = 
                '<div class="error">3D mesh generation failed: ' + (plotJson.error || 'Unknown error') + '</div>';
        }
      } catch (error) {
        console.error('3D mesh generation error:', error);
        document.getElementById('plotly3d').innerHTML = 
            '<div class="error">3D mesh generation failed: ' + error.message + '</div>';
      }
    } else {
      alert('Processing failed: ' + j.error);
    }
  };

  document.getElementById('runDiff').onclick = async () => {
    const filesA = window._uploadedA;
    const filesB = window._uploadedB;
    if (!filesA || !filesB || filesA.length === 0 || filesB.length === 0) {
      alert('Upload both series A and B'); return;
    }

    // Warn if files count is too small for registration
    if (filesA.length < 4 || filesB.length < 4) {
      const proceed = confirm('Warning: One or both series have fewer than 4 slices. Registration may be limited. Continue?');
      if (!proceed) return;
    }

    const payload = {
      files1: filesA,
      files2: filesB,
      do_register: true,
      algorithm: document.getElementById('algorithm').value,
      n_clusters: document.getElementById('n_clusters').value
    };

    // Clear previous results
    document.getElementById('plotly3d').innerHTML = '<p>Processing Differential Analysis...</p>';
    
    const j = await postJSON('/process_differential', payload);
    if (j.success) {
      document.getElementById('previewDiff').innerHTML =
        '<h4>Diff (middle slice)</h4><img src="' + j.diff_slice + '" />' +
        '<h4>Segmented Diff (middle slice)</h4><img src="' + j.seg_slice + '" />';

      // Generate 3D preview from differential with error handling - USE MESH ENDPOINT
      try {
        const resp = await fetch('/generate_3d_mesh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ seg_path: j.seg_path })
        });
        const plotJson = await resp.json();
        console.log('=== DIFFERENTIAL 3D RESPONSE ===');
        console.log('3D Mesh Differential Response:', plotJson);
        console.log('Has mesh:', !!plotJson.mesh);
        console.log('Has meshes:', !!plotJson.meshes);
        console.log('Is array:', Array.isArray(plotJson));
        console.log('Response keys:', Object.keys(plotJson));

        if (plotJson.success) {
          createPlotlyMesh(plotJson, 'plotly3d');  // Pass entire response
        } else {
          document.getElementById('plotly3d').innerHTML = 
            '<div class="error">3D differential mesh generation failed: ' + (plotJson.error || 'Unknown error') + '</div>';
        }
      } catch (error) {
        console.error('3D differential mesh generation error:', error);
        document.getElementById('plotly3d').innerHTML = 
          '<div class="error">3D differential mesh generation failed: ' + error.message + '</div>';
      }
    } else {
      alert('Diff failed: ' + j.error);
    }
  };
});