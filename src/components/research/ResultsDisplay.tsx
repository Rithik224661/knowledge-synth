import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ErrorMessage } from "@/components/ui/error-message";
import { 
  FileText, 
  BarChart3, 
  CheckCircle, 
  Download,
  ExternalLink,
  Star,
  Loader2,
  AlertTriangle
} from "lucide-react";
import "@/styles/visualization.css";
import { useResearch } from "@/contexts/ResearchContext";
import { Message, ResearchResponse } from "@/lib/api";
import { useEffect, useState } from "react";

interface Source {
  title: string;
  url: string;
  type?: "paper" | "article" | "dataset" | string;
  relevanceScore?: number;
  snippet?: string;
}

interface EvaluationMetric {
  metric: string;
  score: number;
  description?: string;
}

interface ResearchResult {
  id: string;
  response: string;
  success: boolean;
  status: 'success' | 'error' | 'processing';
  sources: Source[];
  recommendations: string[];
  evaluation: EvaluationMetric[];
  metadata: {
    queryTime?: number;
    modelUsed?: string;
    tokenCount?: number;
    requestId?: string;
    [key: string]: any;
  };
  timestamp: number;
}

interface ResultsDisplayProps {
  showTabs?: boolean;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ showTabs = false }) => {
  const { messages, isLoading, error, currentResult } = useResearch();
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [activeTab, setActiveTab] = useState('response');

  // Update result when currentResult or messages change
  useEffect(() => {
    console.log('useEffect triggered with currentResult:', currentResult);
    console.log('Messages:', messages);
    
    if (currentResult) {
      console.log('Setting result from currentResult:', currentResult);
      const newResult: ResearchResult = {
        id: currentResult.id,
        response: currentResult.response || 'No response content',
        success: currentResult.success ?? true,
        status: currentResult.status === 'success' || currentResult.status === 'error' || currentResult.status === 'processing' ? currentResult.status : 'success',
        sources: Array.isArray(currentResult.sources) ? currentResult.sources : [],
        recommendations: Array.isArray(currentResult.recommendations) ? currentResult.recommendations : [],
        evaluation: Array.isArray(currentResult.evaluation) ? currentResult.evaluation : [],
        metadata: currentResult.metadata || {},
        timestamp: currentResult.timestamp || Date.now()
      };
      console.log('Setting result state:', newResult);
      setResult(newResult);
    } else {
      // Fall back to the last assistant message if no currentResult
      const assistantMessages = messages.filter(msg => msg.role === 'assistant' && !msg.isError);
      const lastMessage = assistantMessages[assistantMessages.length - 1];
      
      if (lastMessage) {
        console.log('No currentResult, using last assistant message:', lastMessage);
        const metadata = lastMessage.metadata || {};
        const safeMetadata = {
          queryTime: 'queryTime' in metadata ? metadata.queryTime : undefined,
          modelUsed: 'modelUsed' in metadata ? metadata.modelUsed : undefined,
          tokenCount: 'tokenCount' in metadata ? metadata.tokenCount : undefined,
          requestId: 'requestId' in metadata ? metadata.requestId : undefined,
          ...metadata
        };

        const fallbackResult: ResearchResult = {
          id: `msg_${lastMessage.timestamp || Date.now()}`,
          response: lastMessage.content || 'No response content',
          sources: Array.isArray(metadata.sources) ? metadata.sources : [],
          recommendations: Array.isArray(metadata.recommendations) ? metadata.recommendations : [],
          evaluation: Array.isArray(metadata.evaluation) ? metadata.evaluation : [],
          metadata: safeMetadata,
          timestamp: lastMessage.timestamp || Date.now(),
          success: true,
          status: 'success'
        };
        console.log('Setting fallback result:', fallbackResult);
        setResult(fallbackResult);
      } else {
        console.log('No valid messages found, setting result to null');
        setResult(null);
      }
    }
  }, [currentResult, messages]);
  
  // Show loading state with more details
  if (isLoading) {
    return (
      <Card className="p-8 flex flex-col items-center justify-center space-y-4 min-h-[300px]">
        <Loader2 className="h-12 w-12 animate-spin text-primary" />
        <div className="text-center space-y-2">
          <h3 className="text-lg font-medium">Researching your query</h3>
          <p className="text-muted-foreground">This may take a moment...</p>
        </div>
      </Card>
    );
  }
  
  // Show error state
  if (error) {
    return (
      <Card className="p-8 bg-destructive/10 border-destructive">
        <div className="text-destructive flex items-start gap-3">
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="flex-shrink-0 mt-0.5">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <div>
            <h3 className="font-medium">Error Processing Request</h3>
            <p className="text-sm mt-1">{error}</p>
          </div>
        </div>
      </Card>
    );
  }

  // Show empty state if no result
  if (!result) {
    return (
      <Card className="p-8 text-center">
        <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-muted">
          <FileText className="h-6 w-6 text-muted-foreground" />
        </div>
        <h3 className="mt-4 text-lg font-medium">No research results</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Submit a research query to see results here.
        </p>
      </Card>
    );
  }

  // Format the response with enhanced markdown support and better error handling
  const formatResponse = (text: string) => {
    if (!text) {
      return <p className="text-muted-foreground">No response content available</p>;
    }
    
    try {
      // Check if the response contains HTML content (like visualizations)
      if (text.includes('<div') || text.includes('<img') || text.includes('<table')) {
        return (
          <div 
            className="prose prose-sm dark:prose-invert max-w-none" 
            dangerouslySetInnerHTML={{ __html: text }}
          />
        );
      }
      
      // Clean the text: remove markdown separators and standardize formatting
      const cleanedText = text
        .replace(/\s*---+\s*/g, '\n\n') // Remove markdown separators
        .replace(/\s*•\s*--\s*/g, '\n\n') // Remove bullet with dashes
        .replace(/\n\s*•\s*/g, '\n• ') // Standardize bullet points
        .replace(/\n\s*(\d+)\.\s*/g, '\n$1. ') // Standardize numbered lists
        .replace(/\n\s*\[Note:([^\]]*)\]/g, '\n\n*Note: $1*\n\n') // Format notes
        .trim();
      
      // Process text by sections (split by double newlines)
      const sections = cleanedText.split(/\n\s*\n/).filter(Boolean);
      
      return (
        <div className="prose prose-sm dark:prose-invert max-w-none">
          {sections.map((section, i) => {
            // Handle standard markdown headings
            if (section.startsWith('# ')) {
              return <h1 key={i} className="text-2xl font-bold mb-4 text-foreground">{section.substring(2)}</h1>;
            }
            if (section.startsWith('## ')) {
              return <h2 key={i} className="text-xl font-bold mb-3 text-foreground">{section.substring(3)}</h2>;
            }
            if (section.startsWith('### ')) {
              return <h3 key={i} className="text-lg font-bold mb-2 text-foreground">{section.substring(4)}</h3>;
            }
            
            // Handle section titles (e.g., "Research Results")
            if (/^[A-Z][\w\s]+:?$/.test(section.trim())) {
              return <h2 key={i} className="text-xl font-semibold mb-4 text-foreground">{section}</h2>;
            }
            
            // Handle bullet points lists
            if (section.trim().startsWith('•') || section.trim().startsWith('- ')) {
              const items = section.trim().startsWith('•') ?
                section.split(/\n\s*•\s*/).filter(Boolean) :
                section.split(/\n\s*-\s*/).filter(Boolean);
              
              return (
                <ul key={i} className="list-disc pl-5 mb-4 space-y-2">
                  {items.map((item, j) => (
                    <li key={j} className="mb-1">{item.trim()}</li>
                  ))}
                </ul>
              );
            }
            
            // Handle numbered lists
            if (/^\d+\.\s/.test(section.trim())) {
              const items = section.split(/\n\s*\d+\.\s*/).filter(Boolean);
              return (
                <ol key={i} className="list-decimal pl-5 mb-4 space-y-2">
                  {items.map((item, j) => (
                    <li key={j} className="mb-1">{item.trim()}</li>
                  ))}
                </ol>
              );
            }
            
            // Handle headings (lines starting with numbers followed by period)
            if (/^\d+\.\s[A-Z]/.test(section.trim())) {
              const lines = section.split('\n').filter(Boolean);
              const heading = lines[0];
              const content = lines.slice(1).join('\n');
              
              return (
                <div key={i} className="mb-6">
                  <h3 className="text-lg font-medium mb-2 text-foreground">{heading}</h3>
                  {content && (
                    <p className="mb-4">{content}</p>
                  )}
                </div>
              );
            }
            
            // Handle tables if they're in markdown format
            if (section.includes('|') && section.includes('---')) {
              // This is a simplified approach - for complex tables, consider a markdown parser
              return (
                <div key={i} className="table-wrapper mb-6 overflow-x-auto">
                  <table className="min-w-full divide-y divide-border">
                    {section.split('\n').filter(Boolean).map((row, r) => {
                      // Skip rows that are just separator lines
                      if (row.replace(/[\s\-\|]/g, '') === '') return null;
                      
                      const cells = row.split('|').filter(Boolean).map(cell => cell.trim());
                      const isHeaderRow = r === 0 || row.includes('---');
                      
                      // Skip pure separator rows
                      if (cells.every(cell => cell.replace(/\s*-+\s*/g, '') === '')) return null;
                      
                      return (
                        <tr key={r} className={isHeaderRow ? 'bg-muted' : r % 2 === 0 ? 'bg-muted/50' : ''}>
                          {cells.map((cell, c) => {
                            return isHeaderRow ? 
                              <th key={c} className="px-3 py-2 text-left font-medium">{cell.replace(/\s*-+\s*/g, '')}</th> :
                              <td key={c} className="px-3 py-2">{cell}</td>;
                          })}
                        </tr>
                      );
                    }).filter(Boolean)}
                  </table>
                </div>
              );
            }
            
            // Handle key-value pairs (e.g., "Key: Value")
            if (/^[A-Za-z\s]+:\s/.test(section.trim()) && !section.includes('http:') && !section.includes('https:')) {
              const [key, ...valueParts] = section.split(':');
              const value = valueParts.join(':').trim();
              
              return (
                <div key={i} className="mb-4">
                  <span className="font-medium">{key.trim()}:</span> {value}
                </div>
              );
            }
            
            // Default paragraph handling
            return (
              <p key={i} className="mb-4 last:mb-0">
                {section}
              </p>
            );
          })}
        </div>
      );
    } catch (e) {
      console.error('Error formatting response:', e);
      return <p className="text-muted-foreground">{text}</p>;
    }
  };
  
  // Check if the result contains an error
  const hasError = result?.status === 'error' || result?.success === false;

  const getSourceIcon = (type: string) => {
    switch (type) {
      case "paper":
        return <FileText className="h-4 w-4" />;
      case "dataset":
        return <BarChart3 className="h-4 w-4" />;
      default:
        return <ExternalLink className="h-4 w-4" />;
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return "text-success";
    if (score >= 0.6) return "text-warning";
    return "text-destructive";
  };

  // If showTabs is false, we'll only render the content without the tab headers
  const renderContent = () => (
    <Card className="p-6 bg-gradient-card shadow-elevated">
      <div className="space-y-6">
        {hasError && result?.metadata?.error && (
          <ErrorMessage
            title="Research Error"
            message={result.response || "An error occurred during research"}
            details={typeof result.metadata.error === 'object' ? 
              JSON.stringify(result.metadata.error, null, 2) : 
              String(result.metadata.error)}
            icon={<AlertTriangle className="h-4 w-4" />}
          />
        )}
        {showTabs && (
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">Research Results</h2>
            <Button variant="outline" size="sm">
              <Download className="h-4 w-4 mr-2" />
              Export Report
            </Button>
          </div>
        )}

        {showTabs ? (
          <Tabs defaultValue="response" className="w-full">
            <TabsList className="grid w-full grid-cols-5">
              <TabsTrigger value="response">Response</TabsTrigger>
              <TabsTrigger 
                value="visualization" 
                disabled={!(result.response?.includes('<div') || 
                          result.response?.includes('<img') || 
                          result.response?.includes('<table') || 
                          (result.response?.includes('|') && result.response?.includes('---')))}>
                Visualization
              </TabsTrigger>
              <TabsTrigger value="sources" disabled={!result.sources?.length}>
                Sources {result.sources?.length ? `(${result.sources.length})` : ''}
              </TabsTrigger>
              <TabsTrigger value="recommendations" disabled={!result.recommendations?.length}>
                Tips {result.recommendations?.length ? `(${result.recommendations.length})` : ''}
              </TabsTrigger>
              <TabsTrigger value="evaluation" disabled={!result.evaluation?.length}>
                Evaluation
              </TabsTrigger>
            </TabsList>

            <TabsContent value="response" className="space-y-4">
              <Card className="p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  <h3 className="text-lg font-medium mb-4">Research Findings</h3>
                  <div className="space-y-4">
                    {formatResponse(result.response)}
                  </div>
                  {result.metadata?.queryTime && (
                    <div className="mt-4 pt-4 border-t text-sm text-muted-foreground">
                      <p>Query processed in {result.metadata.queryTime.toFixed(2)}s</p>
                      {result.metadata.modelUsed && (
                        <p>Model: {result.metadata.modelUsed}</p>
                      )}
                    </div>
                  )}
                </div>
              </Card>
            </TabsContent>

          <TabsContent value="sources" className="space-y-4">
            {result.sources.length > 0 ? (
              <div className="space-y-4">
                {result.sources.map((source, index) => (
                  <div key={index} className="p-4 border rounded-lg bg-background">
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">
                        {source.type === 'paper' ? (
                          <FileText className="h-5 w-5 text-blue-500" />
                        ) : source.type === 'dataset' ? (
                          <BarChart3 className="h-5 w-5 text-purple-500" />
                        ) : (
                          <ExternalLink className="h-5 w-5 text-green-500" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <a 
                          href={source.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="font-medium text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-1"
                        >
                          {source.title || 'Untitled Source'}
                          <ExternalLink className="h-3.5 w-3.5 ml-0.5 opacity-70" />
                        </a>
                        {source.snippet && (
                          <p className="text-sm text-muted-foreground mt-1">
                            {source.snippet.length > 200 
                              ? `${source.snippet.substring(0, 200)}...` 
                              : source.snippet}
                          </p>
                        )}
                        <div className="flex items-center mt-2 text-xs text-muted-foreground">
                          <span className="capitalize">{source.type || 'source'}</span>
                          {source.relevanceScore !== undefined && (
                            <span className="ml-2 flex items-center">
                              <Star className="h-3 w-3 text-yellow-500 mr-0.5" />
                              {source.relevanceScore.toFixed(1)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <FileText className="mx-auto h-8 w-8 mb-2 opacity-50" />
                <p>No sources available for this query</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="recommendations" className="space-y-4">
            {result.recommendations && result.recommendations.length > 0 ? (
              <div className="space-y-3">
                {result.recommendations.map((rec, index) => (
                  <div key={index} className="flex items-start gap-3 p-4 bg-muted/10 hover:bg-muted/20 rounded-lg transition-colors">
                    <div className="bg-primary/10 p-1.5 rounded-full">
                      <CheckCircle className="h-4 w-4 text-primary" />
                    </div>
                    <p className="text-sm leading-relaxed">{rec}</p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <CheckCircle className="mx-auto h-8 w-8 mb-2 opacity-50" />
                <p>No recommendations available</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="evaluation" className="space-y-4">
            {result.evaluation && result.evaluation.length > 0 ? (
              <div className="grid gap-4">
                {result.evaluation.map((evaluation, index) => (
                  <Card key={index} className="p-4">
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="font-medium text-foreground">{evaluation.metric}</h4>
                      <div className={`text-lg font-bold ${getScoreColor(evaluation.score)}`}>
                        {(evaluation.score * 100).toFixed(0)}%
                      </div>
                    </div>
                    {evaluation.description && (
                      <p className="text-sm text-muted-foreground mb-3">{evaluation.description}</p>
                    )}
                    <div className="w-full bg-muted rounded-full h-2">
                      <div 
                        className={`h-2 rounded-full transition-all ${
                          evaluation.score >= 0.8 ? "bg-success" : 
                          evaluation.score >= 0.6 ? "bg-warning" : "bg-destructive"
                        }`}
                        style={{ width: `${evaluation.score * 100}%` }}
                      />
                    </div>
                  </Card>
                ))}
              </div>
            ) : (
              <div className="p-8 bg-background/50 rounded-lg border text-center">
                <FileText className="h-8 w-8 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">
                  No evaluation metrics were provided for this research.
                </p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="visualization" className="space-y-4">
            {result.response?.includes('<div') || result.response?.includes('<img') || result.response?.includes('<table') || 
             (result.response?.includes('|') && result.response?.includes('---')) ? (
              <Card className="p-6 overflow-x-auto">
                <div className="prose prose-sm dark:prose-invert max-w-none visualization-container">
                  {/* Extract and display markdown tables */}
                  {result.response?.includes('|') && result.response?.includes('---') && (
                    <div className="research-trends-container">
                      <h3>Research Trends Data</h3>
                      <div className="table-wrapper">
                        {result.response
                          .split('\n\n')
                          .filter(section => section.includes('|') && section.includes('---'))
                          .map((table, index) => (
                            <div key={index} className="mb-6">
                              <table className="min-w-full divide-y divide-border">
                                {table.split('\n').filter(Boolean).map((row, r) => {
                                  // Skip rows that are just separator lines
                                  if (row.replace(/[\s\-\|]/g, '') === '') return null;
                                  
                                  const cells = row.split('|').filter(Boolean).map(cell => cell.trim());
                                  const isHeaderRow = r === 0 || row.includes('---');
                                  
                                  // Skip pure separator rows
                                  if (cells.every(cell => cell.replace(/\s*-+\s*/g, '') === '')) return null;
                                  
                                  return (
                                    <tr key={r} className={isHeaderRow ? 'bg-muted' : r % 2 === 0 ? 'bg-muted/50' : ''}>
                                      {cells.map((cell, c) => {
                                        return isHeaderRow ? 
                                          <th key={c} className="px-3 py-2 text-left font-medium">{cell.replace(/\s*-+\s*/g, '')}</th> :
                                          <td key={c} className="px-3 py-2">{cell}</td>;
                                      })}
                                    </tr>
                                  );
                                }).filter(Boolean)}
                              </table>
                            </div>
                          ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Extract and display any HTML visualizations */}
                  {(result.response?.includes('<div') || result.response?.includes('<img') || result.response?.includes('<table')) && (
                    <div dangerouslySetInnerHTML={{ __html: result.response }} />
                  )}
                </div>
              </Card>
            ) : (
              <div className="p-8 bg-background rounded-lg border text-center">
                <BarChart3 className="h-12 w-12 text-muted-foreground mx-auto mb-4" />
                <h3 className="font-medium text-foreground mb-2">No Visualizations Available</h3>
                <p className="text-sm text-muted-foreground">
                  This research query did not generate any visualizations or data tables.
                </p>
              </div>
            )}
          </TabsContent>
        </Tabs>
      ) : (
        // When tabs are hidden, just show the response content
        <Card className="p-6">
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <div className="space-y-4">
              {formatResponse(result.response)}
            </div>
            {result.metadata?.queryTime && (
              <div className="mt-4 pt-4 border-t text-sm text-muted-foreground">
                <p>Query processed in {result.metadata.queryTime.toFixed(2)}s</p>
                {result.metadata.modelUsed && (
                  <p>Model: {result.metadata.modelUsed}</p>
                )}
              </div>
            )}
          </div>
        </Card>
      )}
    </div>
  </Card>
);

return renderContent();
};