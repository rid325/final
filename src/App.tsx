import { useState } from 'react';
import { UploadZone } from './components/UploadZone';
import { SplitView } from './components/SplitView';
import { PDFViewer } from './components/PDFViewer';
import { DataGrid } from './components/DataGrid';
import { JSONViewer } from './components/JSONViewer';
import { SummaryWidget } from './components/SummaryWidget';
import { ExportButtons } from './components/ExportButtons';
import { MOCK_BOUNDING_BOXES, MOCK_EXTRACTED_DATA, MOCK_INSIGHTS } from './lib/mockData';
import type { ExtractedData, BoundingBox } from './lib/appTypes';
import { Layout, FileText, Table, Code, Loader2, Download } from 'lucide-react';
import styles from './App.module.css';

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [viewMode, setViewMode] = useState<'landing' | 'workspace'>('landing');
  const [rightPaneView, setRightPaneView] = useState<'grid' | 'json'>('grid');
  const [selectedRegionId, setSelectedRegionId] = useState<string | undefined>(undefined);
  const [highlightedRowIndex, setHighlightedRowIndex] = useState<number | undefined>(undefined);

  const [extractedData, setExtractedData] = useState<ExtractedData>(MOCK_EXTRACTED_DATA);
  const [boundingBoxes, setBoundingBoxes] = useState<BoundingBox[]>(MOCK_BOUNDING_BOXES);
  const [aiInsights, setAiInsights] = useState<any>(MOCK_INSIGHTS);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleUploadComplete = async (uploadedFile: File) => {
    setFile(uploadedFile);
    setIsAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);
    formData.append('use_ai', 'true'); // Default to true for this demo

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const result = await response.json();

      // Map backend response to frontend types
      const tables = result.ai_insights?.tables || [];
      const firstTable = tables[0] || { headers: [], rows: [] };

      // Transform headers to AG Grid column definitions
      const columns = firstTable.headers.map((header: string) => ({
        field: header,
        headerName: header
      }));

      // Transform rows to objects keyed by header
      const rows = firstTable.rows.map((row: string[], index: number) => {
        const rowObj: any = { id: index };
        firstTable.headers.forEach((header: string, i: number) => {
          rowObj[header] = row[i];
        });
        return rowObj;
      });

      const mappedData: ExtractedData = {
        columns: columns,
        rows: rows,
        tables: tables,
        keyValuePairs: result.ai_insights?.key_value_pairs || {},
        text: result.analysis?.pages?.[0]?.text?.join('\n') || result.analysis?.text?.join('\n') || '',
        errors: []
      };

      // Parse Bounding Boxes
      const newBoundingBoxes: BoundingBox[] = [];
      const analysis = result.analysis;

      if (analysis) {
        if (analysis.type === 'pdf' && analysis.pages) {
           analysis.pages.forEach((page: any) => {
               const pageNum = page.page;
               // Text Blocks
               if (page.blocks) {
                   page.blocks.forEach((block: any) => {
                       newBoundingBoxes.push({
                           id: block.id,
                           page: pageNum,
                           type: 'text',
                           coordinates: block.normalized_box,
                           label: block.text.substring(0, 20) + '...'
                       });
                   });
               }
               // Tables
               if (page.tables) {
                   page.tables.forEach((table: any) => {
                        newBoundingBoxes.push({
                           id: table.id,
                           page: pageNum,
                           type: 'table',
                           coordinates: table.normalized_box,
                           label: 'Table'
                       });
                   });
               }
               // Charts
               if (page.charts) {
                   page.charts.forEach((chart: any) => {
                        newBoundingBoxes.push({
                           id: chart.id,
                           page: pageNum,
                           type: chart.type === 'pie' ? 'chart' : 'chart',
                           coordinates: chart.normalized_box,
                           label: chart.type === 'pie' ? 'Pie Chart' : 'Chart'
                       });
                   });
               }
           });
        } else if (analysis.type === 'image') {
             // Image logic
             if (analysis.blocks) {
                 analysis.blocks.forEach((block: any) => {
                    newBoundingBoxes.push({
                        id: block.id,
                        page: 1,
                        type: 'text',
                        coordinates: block.normalized_box,
                        label: block.text.substring(0, 20) + '...'
                    });
                 });
             }
             if (analysis.tables) {
                analysis.tables.forEach((table: any) => {
                    newBoundingBoxes.push({
                        id: table.id,
                        page: 1,
                        type: 'table',
                        coordinates: table.normalized_box,
                        label: 'Table'
                    });
                });
             }
             if (analysis.charts) {
                analysis.charts.forEach((chart: any) => {
                    newBoundingBoxes.push({
                        id: chart.id,
                        page: 1,
                        type: 'chart',
                        coordinates: chart.normalized_box,
                        label: chart.type === 'pie' ? 'Pie Chart' : 'Chart'
                    });
                });
             }
        }
      }

      // Transform AI result into SummaryInsight array
      const insights: any[] = [];

      if (result.ai_insights?.summary) {
        insights.push({
          id: 'summary-1',
          type: 'summary',
          content: result.ai_insights.summary
        });
      }

      if (result.ai_insights?.topics) {
        result.ai_insights.topics.forEach((topic: string, idx: number) => {
          insights.push({
            id: `topic-${idx}`,
            type: 'trend', // Using 'trend' as a generic type for topics
            content: topic
          });
        });
      }

      setExtractedData(mappedData);
      setBoundingBoxes(newBoundingBoxes);
      setAiInsights(insights);
      setDownloadUrl(result.downloadUrl);
      setViewMode('workspace');
    } catch (err) {
      console.error(err);
      setError('Failed to analyze document. Please try again.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleRegionClick = (regionId: string) => {
    setSelectedRegionId(regionId);

    // Simple mapping: if text, try to find row.
    // For now, we don't have a direct map from text-block to table-row.
    // But we can check if the region is a table.

    if (regionId.startsWith('table-')) {
         // If a table is selected, maybe highlight the first row or all?
         // For now, just highlight the first row to show interaction
         setHighlightedRowIndex(0);
    } else {
         setHighlightedRowIndex(undefined);
    }
  };

  const renderLeftPane = () => (
    file ? (
      <PDFViewer
        file={file}
        boundingBoxes={boundingBoxes}
        selectedRegionId={selectedRegionId}
        onRegionClick={handleRegionClick}
      />
    ) : null
  );

  const renderRightPane = () => (
    <div className={styles.rightPaneContainer}>
      <div className={styles.tabs}>
        <div className={styles.tabGroup}>
          <button
            className={`${styles.tab} ${rightPaneView === 'grid' ? styles.activeTab : ''}`}
            onClick={() => setRightPaneView('grid')}
          >
            <Table size={16} />
            Grid View
          </button>
          <button
            className={`${styles.tab} ${rightPaneView === 'json' ? styles.activeTab : ''}`}
            onClick={() => setRightPaneView('json')}
          >
            <Code size={16} />
            JSON View
          </button>
        </div>
        {downloadUrl && (
          <a href={downloadUrl} className={styles.downloadLink} download>
            <Download size={16} />
            Download Full Report
          </a>
        )}
      </div>

      <div className={styles.contentArea}>
        <div className={styles.mainContent}>
          {rightPaneView === 'grid' ? (
            <DataGrid
              data={extractedData}
              highlightedRowIndex={highlightedRowIndex}
            />
          ) : (
            <JSONViewer data={extractedData} />
          )}
        </div>
        <div className={styles.summaryPanel}>
          <SummaryWidget insights={aiInsights} />
        </div>
      </div>
    </div>
  );

  return (
    <div className={styles.app}>
      <header className={styles.header}>
        <div className={styles.logo}>
          <Layout className={styles.logoIcon} />
          <h1 className={styles.title}>DataExtractor<span className={styles.highlight}>Pro</span></h1>
        </div>

        {viewMode === 'workspace' && (
          <div className={styles.actions}>
            <div className={styles.fileInfo}>
              <FileText size={16} />
              <span>{file?.name}</span>
            </div>
            <div className={styles.divider} />
            <ExportButtons data={extractedData} />
          </div>
        )}
      </header>

      <main className={styles.main}>
        {viewMode === 'landing' ? (
          <div className={styles.landing}>
            <div className={styles.hero}>
              <h2 className={styles.heroTitle}>
                Transform Documents into <br />
                <span className={styles.gradientText}>Actionable Data</span>
              </h2>
              <p className={styles.heroSubtitle}>
                Upload PDFs or images. Verify extraction visually. Export clean data.
              </p>
            </div>

            {isAnalyzing ? (
              <div className={styles.loadingState}>
                <Loader2 className={styles.spinner} size={48} />
                <p>Analyzing document with AI...</p>
              </div>
            ) : (
              <UploadZone onUploadComplete={handleUploadComplete} />
            )}

            {error && <div className={styles.error}>{error}</div>}

            <div className={styles.features}>
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>1</div>
                <h3>Upload</h3>
                <p>Drag & drop complex reports</p>
              </div>
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>2</div>
                <h3>Extract</h3>
                <p>AI identifies tables & charts</p>
              </div>
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>3</div>
                <h3>Verify</h3>
                <p>Visual side-by-side check</p>
              </div>
              <div className={styles.featureCard}>
                <div className={styles.featureIcon}>4</div>
                <h3>Export</h3>
                <p>Download clean Excel/JSON</p>
              </div>
            </div>
          </div>
        ) : (
          <SplitView
            leftPane={renderLeftPane()}
            rightPane={renderRightPane()}
            initialLeftWidth={45}
          />
        )}
      </main>
    </div>
  );
}

export default App;
