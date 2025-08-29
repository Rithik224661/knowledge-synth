import { useState, useRef, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Search, Sparkles, Copy, Loader2, ExternalLink, FileText, Upload, X } from "lucide-react";
import { useResearch } from "@/contexts/ResearchContext";
import { toast } from "sonner";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

const SourceCard = ({ source, index }: { source: any; index: number }) => (
  <a
    key={index}
    href={source.url}
    target="_blank"
    rel="noopener noreferrer"
    className="block p-3 border rounded-lg hover:bg-muted/30 transition-colors"
  >
    <div className="flex items-start gap-3">
      <div className="mt-0.5">
        {source.type === 'paper' ? 
          <FileText className="h-4 w-4 text-blue-500" /> : 
          <ExternalLink className="h-4 w-4 text-green-500" />
        }
      </div>
      <div className="flex-1 min-w-0">
        <div className="font-medium text-foreground hover:underline flex items-center gap-1">
          {source.title || 'Untitled Source'}
          <ExternalLink className="h-3 w-3 opacity-70 ml-1" />
        </div>
        {source.snippet && (
          <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
            {source.snippet}
          </p>
        )}
      </div>
    </div>
  </a>
);

export const QueryInput = () => {
  const [query, setQuery] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [activeTab, setActiveTab] = useState<string>("query");
  const { submitQuery, submitFileQuery, isLoading, messages, sources, currentResult } = useResearch();
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const lastSubmissionTime = useRef<number>(0);
  const resultsContainerRef = useRef<HTMLDivElement>(null);
  
  const handleSubmit = async () => {
    const now = Date.now();
    
    // Prevent rapid submissions
    if (now - lastSubmissionTime.current < 2000) {
      toast.warning('Please wait a moment before submitting again');
      return;
    }
    
    // Input validation
    const trimmedQuery = query.trim();
    if (!trimmedQuery) {
      toast.error('Please enter a research query');
      return;
    }
    
    if (trimmedQuery.length > 1000) {
      toast.warning('Query is too long. Please limit to 1000 characters');
      return;
    }
    
    try {
      setIsSubmitting(true);
      lastSubmissionTime.current = now;
      
      if (activeTab === "file" && selectedFile) {
        // Submit query with file
        await submitFileQuery(trimmedQuery, selectedFile);
        setSelectedFile(null);
      } else {
        // Submit regular query
        await submitQuery(trimmedQuery);
      }
      
      setQuery("");
      // Focus the textarea again after successful submission
      setTimeout(() => textareaRef.current?.focus(), 100);
    } catch (error) {
      // Extract error details if available
      const errorMessage = error instanceof Error ? error.message : 'Failed to submit query';
      
      // Check for detailed error information
      let errorDetails = null;
      if (error instanceof Error && (error as any).details) {
        errorDetails = (error as any).details;
      }
      
      // Display error toast with more details
      if (errorDetails) {
        toast.error(errorMessage, {
          description: typeof errorDetails === 'object' ? 
            JSON.stringify(errorDetails).substring(0, 100) + '...' : 
            String(errorDetails).substring(0, 100) + '...'
        });
      } else {
        toast.error(errorMessage);
      }
    } finally {
      setIsSubmitting(false);
      // Clear the file input after submission
      if (activeTab === "file") {
        clearSelectedFile();
      }
    }
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    setSelectedFile(file);
  };
  
  const clearSelectedFile = () => {
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const exampleQueries = [
    "Summarize recent advances in AI safety",
    "Find papers on multimodal AI from the last 6 months",
    "Research federated learning in healthcare"
  ];

  const handleKeyPress = (e: React.KeyboardEvent) => {
    // Submit on Cmd+Enter or Ctrl+Enter
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      handleSubmit();
    }
  };
  
  const renderFileUploadUI = () => {
    return (
      <div className="space-y-4">
        <div className="flex flex-col space-y-2">
          <Label htmlFor="file-upload" className="text-sm font-medium">
            Upload Document (PDF, DOCX, TXT)
          </Label>
          <div className="flex items-center gap-2">
            <Input
              id="file-upload"
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx,.doc,.txt"
              onChange={handleFileChange}
              className="flex-1"
              disabled={isLoading}
            />
            {selectedFile && (
              <Button
                variant="ghost"
                size="icon"
                onClick={clearSelectedFile}
                disabled={isLoading}
              >
                <X className="h-4 w-4" />
              </Button>
            )}
          </div>
          {selectedFile && (
            <p className="text-xs text-muted-foreground mt-1">
              Selected: {selectedFile.name} ({Math.round(selectedFile.size / 1024)} KB)
            </p>
          )}
        </div>
      </div>
    );
  };
  
  // Auto-resize textarea as user types
  const adjustTextareaHeight = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  };
  
  useEffect(() => {
    adjustTextareaHeight();
  }, [query]);

  // Scroll to results when they're available
  useEffect(() => {
    if (currentResult && resultsContainerRef.current) {
      resultsContainerRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [currentResult]);

  // Toggle results display when we have results
  useEffect(() => {
    if (currentResult) {
      setShowResults(true);
    }
  }, [currentResult]);

  // Find the latest assistant message with content
  const latestMessage = [...messages].reverse().find(msg => 
    msg.role === 'assistant' && msg.content
  );

  // Format the response content with proper spacing and sections
  const formatResponse = (text: string) => {
    if (!text) return null;
    
    // First, clean the text
    let cleanText = text
      // Remove markdown headers but preserve section breaks
      .replace(/^#+\s*(.*?)\s*#*$/gm, '\n$1\n')
      // Remove markdown bold/italic but keep the text
      .replace(/(\*\*|__|\*|_)(.*?)\1/g, '$2')
      // Remove markdown links but keep the text
      .replace(/\[(.*?)\]\(.*?\)/g, '$1')
      // Remove HTML tags
      .replace(/<[^>]*>/g, '')
      // Remove watermarks and signatures
      .replace(/\*?\s*(generated|created|written)\s*(by|with|using).*?\*?/gi, '')
      .replace(/\*?\s*Powered by.*?\*?/gi, '')
      // Normalize multiple newlines to double newlines for paragraphs
      .replace(/\n{3,}/g, '\n\n')
      // Clean up any remaining markdown artifacts
      .replace(/`{3,}.*?`{3,}/gs, '') // Remove code blocks
      .replace(/`/g, '') // Remove inline code markers
      .replace(/^\s*[-*+]\s*/gm, '• ') // Convert markdown lists to bullet points
      .trim();

    // Split into paragraphs and process each one
    const paragraphs = cleanText.split('\n\n');
    
    return (
      <div className="space-y-6">
        {paragraphs.map((paragraph, i) => {
          const trimmed = paragraph.trim();
          if (!trimmed) return null;
          
          // Check if this looks like a heading (short text, ends with colon, or is in title case)
          const isHeading = trimmed.length < 60 && 
                           (trimmed.endsWith(':') || 
                            trimmed === trimmed.toUpperCase() ||
                            (trimmed.split(' ').length < 5 && trimmed === trimmed[0].toUpperCase() + trimmed.slice(1).toLowerCase()));
          
          return (
            <div key={i} className={isHeading ? 'mb-2' : 'mb-4'}>
              {isHeading ? (
                <h3 className="text-lg font-semibold text-foreground mb-3 pt-4 first:pt-0 border-t border-border/50 first:border-t-0">
                  {trimmed.replace(/:$/, '')}
                </h3>
              ) : (
                <p className="text-foreground/90 leading-relaxed">
                  {trimmed}
                </p>
              )}
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      <Card className="p-6 sm:p-8 card-elevated">
        <div className="space-y-6">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-gradient-research rounded-lg flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-5 w-5 text-primary-foreground" />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-foreground">Research Assistant</h2>
              <p className="text-sm text-muted-foreground mt-1">Get comprehensive research on any topic</p>
            </div>
          </div>
          
          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="grid w-full grid-cols-2 mb-4">
              <TabsTrigger value="query" disabled={isLoading || isSubmitting}>
                <Search className="h-4 w-4 mr-2" />
                Text Query
              </TabsTrigger>
              <TabsTrigger value="file" disabled={isLoading || isSubmitting}>
                <Upload className="h-4 w-4 mr-2" />
                Upload Document
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="query" className="space-y-4">
              <div className="relative">
                <Textarea
                  ref={textareaRef}
                  placeholder="Ask me to research, analyze, and synthesize information from multiple sources..."
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    adjustTextareaHeight();
                  }}
                  onKeyDown={handleKeyPress}
                  className="min-h-[120px] max-h-[200px] resize-none border-border/60 focus:ring-2 focus:ring-primary/20 focus:border-primary text-base leading-relaxed pr-12"
                  disabled={isLoading || isSubmitting}
                />
                <div className="absolute bottom-3 right-3 flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {query.length}/1000
                  </span>
                  <Button 
                    size="sm" 
                    onClick={handleSubmit}
                    disabled={isLoading || isSubmitting || !query.trim()}
                    className="h-8 w-8 p-0 rounded-full"
                  >
                    {isLoading || isSubmitting ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Search className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>
            </TabsContent>
            
            <TabsContent value="file" className="space-y-4">
              {renderFileUploadUI()}
              <div className="relative">
                <Textarea
                  placeholder="Ask a question about the uploaded document..."
                  value={query}
                  onChange={(e) => {
                    setQuery(e.target.value);
                    adjustTextareaHeight();
                  }}
                  onKeyDown={handleKeyPress}
                  disabled={isLoading || isSubmitting}
                  className="min-h-[120px] max-h-[200px] resize-none border-border/60 focus:ring-2 focus:ring-primary/20 focus:border-primary text-base leading-relaxed pr-12"
                />
                <div className="absolute bottom-3 right-3 flex items-center gap-2">
                  <span className="text-xs text-muted-foreground">
                    {query.length}/1000
                  </span>
                </div>
              </div>
            </TabsContent>
          </Tabs>
          
          <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 pt-1">
            <div className="text-xs text-muted-foreground">
              <p>Press <kbd className="px-1.5 py-0.5 bg-muted rounded text-[0.7rem] font-mono border border-border">⌘ + Enter</kbd> to submit</p>
            </div>
            <Button 
              onClick={handleSubmit}
              disabled={isLoading || isSubmitting || !query.trim() || (activeTab === "file" && !selectedFile)}
              className="w-full sm:w-auto gap-2"
              size="lg"
            >
              {isLoading || isSubmitting ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Researching...
                </>
              ) : (
                <>
                  <Search className="h-4 w-4" />
                  Start Research
                </>
              )}
            </Button>
          </div>
          
          <div className="pt-6 border-t border-border/30">
            <h3 className="text-sm font-medium text-muted-foreground mb-3">Try these example queries:</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              {exampleQueries.map((example, i) => (
                <button
                  key={i}
                  onClick={() => {
                    setQuery(example);
                    setTimeout(() => textareaRef.current?.focus(), 100);
                  }}
                  className="text-left p-3 text-sm border rounded-lg hover:bg-muted/30 transition-colors text-muted-foreground hover:text-foreground flex items-start gap-2"
                  disabled={isLoading || isSubmitting}
                >
                  <Copy className="h-3 w-3 mt-0.5 flex-shrink-0" />
                  <span>{example}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Results Section */}
        {showResults && (currentResult || latestMessage) && (
          <div 
            ref={resultsContainerRef}
            className="mt-8 pt-6 border-t border-border/30"
          >
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Research Results
            </h3>
            
            <div className="space-y-6">
              {/* Main Response */}
              <Card className="p-6">
                <div className="prose prose-sm dark:prose-invert max-w-none">
                  {formatResponse(currentResult?.response || latestMessage?.content || 'No response content')}
                </div>
                
                {/* Metadata */}
                {currentResult?.metadata?.queryTime && (
                  <div className="mt-6 pt-4 border-t text-sm text-muted-foreground">
                    <div className="flex items-center space-x-2">
                      <span>⏱️ Processed in {currentResult.metadata.queryTime.toFixed(2)}s</span>
                      {currentResult.metadata.modelUsed && (
                        <span>• 🧠 Model: {currentResult.metadata.modelUsed}</span>
                      )}
                    </div>
                  </div>
                )}
              </Card>

              {/* Sources */}
              {sources && sources.length > 0 && (
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-muted-foreground">Sources</h4>
                  <div className="space-y-3">
                    {sources.map((source: any, index: number) => (
                      <SourceCard key={index} source={source} index={index} />
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </Card>
    </div>
  );
};