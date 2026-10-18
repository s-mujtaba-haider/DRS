import { Suspense } from "react";

import UmpireMonitor from "@/components/UmpireMonitor";
import { Empty } from "@/components/ui";

async function Loader({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <UmpireMonitor id={id} />;
}

export default function Page({ params }: { params: Promise<{ id: string }> }) {
  return (
    <Suspense fallback={<Empty>Loading monitor...</Empty>}>
      <Loader params={params} />
    </Suspense>
  );
}
