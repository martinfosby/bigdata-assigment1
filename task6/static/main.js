// =======================
// Global variables
// =======================
let currentSegmentationData = null;
let currentAlgorithmResults = {};
let current3DViewer = null;

// Color palette for clusters (if needed as fallback)
const CLUSTER_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFBE0B', '#FB5607',
    '#8338EC', '#3A86FF', '#38B000', '#F15BB5', '#9B5DE5',
    '#FEE440', '#00BBF9', '#00F5D4', '#FF9E00', '#E63946'
];

// =======================
// Core utility functions
// =======================
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
    else { alert('Upload failed: ' + (j.error || 'Unknown error')); return []; }
}

// =======================
// UI setup and handling
// =======================
function updateAlgorithmParameters() {
    const algorithm = document.getElementById('algorithm').value;
    
    // Show/hide appropriate parameter fields
    if (algorithm === 'dbscan') {
        document.getElementById('dbscanParams').style.display = 'block';
        document.getElementById('kmeansParams').style.display = 'none';
    } else {
        document.getElementById('dbscanParams').style.display = 'none';
        document.getElementById('kmeansParams').style.display = 'block';
    }
}

function createPlotlyPointCloud(responseData, containerId) {
    console.log('Creating 3D point cloud visualization with data:', responseData);

    if (!responseData.point_clouds || responseData.point_clouds.length === 0) {
        document.getElementById(containerId).innerHTML = 
            '<div class="error">No point cloud data available for visualization</div>';
        return;
    }
    
    const traces = [];

    responseData.point_clouds.forEach((cluster) => {
        const [r, g, b, a] = cluster.color;
        const colorStr = `rgba(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)},${a})`;

        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: cluster.x,
            y: cluster.y,
            z: cluster.z,
            marker: {
                size: 2,
                color: colorStr,
                opacity: 0.7
            },
            name: `Cluster ${cluster.cluster_id} (${cluster.point_count} points)`,
            hoverinfo: 'name'
        });
    });

    const layout = {
        title: `3D Point Cloud Segmentation (${responseData.point_clouds.length} clusters, ${responseData.stats.total_points_displayed} points)`,
        scene: {
            xaxis: { title: 'X' },
            yaxis: { title: 'Y' },
            zaxis: { title: 'Z' }
        },
        showlegend: true
    };

    Plotly.newPlot(containerId, traces, layout);


    
    // Update stats display
    const stats = responseData.stats;
    const statsHtml = `
        <div style="margin-top: 20px; padding: 10px; background: #f5f5f5; border-radius: 5px;">
            <h4>Statistics:</h4>
            <p>Total clusters: ${stats.n_clusters}</p>
            <p>Total voxels: ${stats.total_voxels.toLocaleString()}</p>
            <p>Segmented voxels: ${stats.segmented_voxels.toLocaleString()}</p>
            <p>Points displayed: ${stats.total_points_displayed.toLocaleString()}</p>
        </div>
    `;
    document.getElementById('plotStats').innerHTML = statsHtml;
}

// =======================
// Processing logic
// =======================
async function handleFileUpload() {
    const filesInput = document.getElementById('filesSingle');
    const uploaded = await uploadFiles(filesInput);
    window._uploadedSingle = uploaded;
    alert('Uploaded ' + uploaded.length + ' files');
}

async function processSingleSeries() {
    const files = window._uploadedSingle;
    if (!files || files.length === 0) {
        alert('Please upload files first');
        return;
    }

    const payload = {
        files: files,
        algorithm: document.getElementById('algorithm').value,
        n_clusters: parseInt(document.getElementById('n_clusters').value),
        eps: parseFloat(document.getElementById('eps').value),
        min_samples: parseInt(document.getElementById('min_samples').value),
        use_supervoxels: document.getElementById('use_supervoxels').checked,
        supervoxel_count: parseInt(document.getElementById('supervoxel_count').value),
        spatial_weight: parseFloat(document.getElementById('spatial_weight').value),
        intensity_weight: parseFloat(document.getElementById('intensity_weight').value),
        use_pca: document.getElementById('use_pca').checked,
        pca_components: parseInt(document.getElementById('pca_components').value)
    };

    document.getElementById('plotly3d').innerHTML = '<p>Processing Series...</p>';
    
    try {
        const j = await postJSON('/process_series', payload);
        if (j.success) {
            document.getElementById('previewOrig').innerHTML =
                '<h4>Original (middle slice)</h4><img src="' + j.orig_slice + '" />';
            document.getElementById('previewSeg').innerHTML =
                '<h4>Segmented (middle slice)</h4><img src="' + j.seg_slice + '" />';
            window._seg_path = j.seg_path;

            const meshResponse = await postJSON('/generate_3d_mesh', { seg_path: j.seg_path });
            if (meshResponse.success) {
                createPlotlyPointCloud(meshResponse, 'plotly3d');
            } else {
                document.getElementById('plotly3d').innerHTML = 
                    '<div class="error">3D mesh generation failed: ' + (meshResponse.error || 'Unknown error') + '</div>';
            }
        } else {
            alert('Processing failed: ' + j.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Processing failed: ' + error.message);
    }
}

async function processDifferential() {
    const filesA = window._uploadedA;
    const filesB = window._uploadedB;
    if (!filesA || !filesB || filesA.length === 0 || filesB.length === 0) {
        alert('Upload both series A and B first');
        return;
    }

    const payload = {
        files1: filesA,
        files2: filesB,
        do_register: document.getElementById('do_register').checked,
        algorithm: document.getElementById('diffAlgorithm').value,
        n_clusters: parseInt(document.getElementById('diffNClusters').value)
    };

    document.getElementById('plotly3d').innerHTML = '<p>Processing Differential Analysis...</p>';
    
    try {
        const j = await postJSON('/process_differential', payload);
        if (j.success) {
            document.getElementById('previewDiff').innerHTML =
                '<h4>Difference (middle slice)</h4><img src="' + j.diff_slice + '" />' +
                '<h4>Segmented Difference (middle slice)</h4><img src="' + j.seg_slice + '" />';

            const meshResponse = await postJSON('/generate_3d_mesh', { seg_path: j.seg_path });
            if (meshResponse.success) {
                createPlotlyPointCloud(meshResponse, 'plotly3d');
            } else {
                document.getElementById('plotly3d').innerHTML = 
                    '<div class="error">3D differential mesh generation failed: ' + (meshResponse.error || 'Unknown error') + '</div>';
            }
        } else {
            alert('Differential analysis failed: ' + j.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Differential analysis failed: ' + error.message);
    }
}

// =======================
// Initialization
// =======================
function setupEventListeners() {
    document.getElementById('algorithm').addEventListener('change', updateAlgorithmParameters);
    document.getElementById('uploadSingleBtn').addEventListener('click', handleFileUpload);
    document.getElementById('runSingleBtn').addEventListener('click', processSingleSeries);
    document.getElementById('runDiffBtn').addEventListener('click', processDifferential);
}

document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateAlgorithmParameters();
});
