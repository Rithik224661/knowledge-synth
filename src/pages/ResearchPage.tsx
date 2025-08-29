import { QueryInput } from "@/components/research/QueryInput";
import { Card } from "@/components/ui/card";

export default function ResearchPage() {
  return (
    <div className="container mx-auto px-4 py-6 md:py-8">
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI Research Assistant</h1>
          <p className="text-muted-foreground">
            Get comprehensive research on any topic with AI-powered analysis
          </p>
        </div>

        <div className="grid gap-6">
          {/* Main content - now full width */}
          <div className="w-full max-w-4xl mx-auto">
            <QueryInput />
          </div>
        </div>
      </div>
    </div>
  );
}
